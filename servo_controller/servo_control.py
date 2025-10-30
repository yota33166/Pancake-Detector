"""Servo control logic for the pancake dispenser running on Raspberry Pi.

This module mirrors the behavior of the supplied Arduino sketch by:

* Driving two hobby servos attached to the configured GPIO pins
* Reading four buttons with internal pull-ups to control servo motions
* Toggling an LED that indicates manual mode is active
* Providing an automatic pour cycle when manual mode is off
* Emitting debug timing information for how long button 1 remains pressed

The implementation leverages the :mod:`gpiozero` library and should be
executed on an actual Raspberry Pi. When run on another platform, import
errors will be surfaced with a descriptive message.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
import queue
from dataclasses import dataclass
from typing import Dict, Optional

from gpiozero import AngularServo, Button, LED, exc  # type: ignore[misc]


@dataclass(frozen=True)
class PinConfig:
	"""GPIOピン割り当てを保持する設定。"""

	servo_left: int = 12
	servo_right: int = 13
	button_pour: int = 17
	button_mode: int = 27
	button_left: int = 22
	button_right: int = 23
	led: Optional[int] = None


@dataclass(frozen=True)
class ServoPulseConfig:
	"""サーボ駆動用のPWM関連設定を保持する。"""

	pwm_frequency: int = 50  # Hertz (spec range 50-330 Hz)
	min_pulse_us: int = 500
	max_pulse_us: int = 2500


@dataclass(frozen=True)
class MotionConfig:
	"""サーボの角度とタイミング設定をまとめる。"""

	rotate_delay_ms: int = 20
	pour_time_ms: int = 1000
	max_angle: int = 180
	start_angle_left: int = 160
	end_angle_left: int = 100
	start_angle_right: int = 20
	end_angle_right: int = 80
	step_delay_s: float = 0.005
	reverse_left: bool = True
	reverse_right: bool = True


DEFAULT_DEBUG = True
DEFAULT_PINS = PinConfig()
DEFAULT_PULSE = ServoPulseConfig()
DEFAULT_MOTION = MotionConfig()
DEFAULT_DEBOUNCE_DELAY_S = 0.01


def clamp(value: int, lower: int, upper: int) -> int:
	"""最小値と最大値の範囲にvalueをクランプする。"""

	return max(lower, min(upper, value))


@dataclass
class ServoConfig:
	"""サーボの設定と状態を保持する。
	Parameters:
		servo: gpiozeroのAngularServoインスタンス。
		id: サーボの一意の識別子。(left, rightなど)
		pin: サーボが接続されているGPIOピン番号。
		start_angle: 回転の"開始"位置を表す角度。
		end_angle: 回転の"終了"位置を表す角度。
		current_angle: サーボの最後の角度。
	"""

	servo: AngularServo
	id: str
	pin: int
	start_angle: int
	end_angle: int
	current_angle: int
	is_reversed: bool = True


class ServoController:
	"""サーボとGPIOデバイスのライフサイクルを管理するクラス。
	
	Args:
		command_queue: コマンドを送信するキュー。
		pins: GPIOピン割り当て設定。
		pulse: サーボPWM設定。
		motion: サーボ動作設定。
		debounce_delay_s: ボタンデバウンス時間（秒）。
		debug: デバッグログを有効にするかどうか。
	"""

	def __init__(
		self,
		command_queue: Optional[queue.Queue] = None,
		*,
		pins: PinConfig = DEFAULT_PINS,
		pulse: ServoPulseConfig = DEFAULT_PULSE,
		motion: MotionConfig = DEFAULT_MOTION,
		debounce_delay_s: float = DEFAULT_DEBOUNCE_DELAY_S,
		debug: bool = DEFAULT_DEBUG,
	) -> None:
		self.command_queue = command_queue
		self.pins = pins
		self.pulse = pulse
		self.motion = motion
		self.debounce_delay_s = debounce_delay_s
		self.debug = debug

		self.servo_states: Dict[str, ServoConfig] = {}
		self.pour_hold_active: Dict[str, bool] = {"left": False, "right": False}
		self.is_manual: bool = False
		self.press_start_time: Optional[float] = None

		self.button_pour: Optional[Button] = None
		self.button_mode: Optional[Button] = None
		self.button_left: Optional[Button] = None
		self.button_right: Optional[Button] = None
		self.led: Optional[LED] = None

	def _setup_logging(self) -> None:
		"""ログの設定を行う"""

		level = logging.DEBUG if self.debug else logging.INFO
		logging.basicConfig(
			level=level,
			format="[%(asctime)s] %(levelname)s: %(message)s",
			datefmt="%H:%M:%S",
		)

	def _setup_devices(self) -> None:
		"""サーボ，ボタン，LEDの初期化を行う．"""

		self.button_pour = Button(self.pins.button_pour, pull_up=True)
		self.button_mode = Button(self.pins.button_mode, pull_up=True)
		self.button_left = Button(self.pins.button_left, pull_up=True)
		self.button_right = Button(self.pins.button_right, pull_up=True)

		if self.pins.led is not None:
			self.led = LED(self.pins.led)
			self.led.off()

		left_state = self._create_servo_state(
			servo_id="left",
			pin=self.pins.servo_left,
			start_angle=self.motion.start_angle_left,
			end_angle=self.motion.end_angle_left,
			is_reversed=self.motion.reverse_left,
		)
		right_state = self._create_servo_state(
			servo_id="right",
			pin=self.pins.servo_right,
			start_angle=self.motion.start_angle_right,
			end_angle=self.motion.end_angle_right,
			is_reversed=self.motion.reverse_right,
		)

		self.servo_states = {
			left_state.id: left_state,
			right_state.id: right_state,
		}

	def _create_servo_state(
		self,
		*,
		servo_id: str,
		pin: int,
		start_angle: int,
		end_angle: int,
		is_reversed: bool,
	) -> ServoConfig:
		"""初期化されたサーボオブジェクトを作成し、ServoConfigを返す。
		
		Args:
			servo_id: サーボの一意の識別子。
			pin: サーボが接続されているGPIOピン番号。
			start_angle: 回転の"開始"位置を表す角度。
			end_angle: 回転の"終了"位置を表す角度。
			is_reversed: サーボの回転方向が反転しているかどうか。
		"""

		logical_initial = clamp(start_angle, 0, self.motion.max_angle)
		physical_initial = (
			self.motion.max_angle - logical_initial if is_reversed else logical_initial
		)
		servo = AngularServo(
			pin,
			min_angle=0,
			max_angle=self.motion.max_angle,
			min_pulse_width=self.pulse.min_pulse_us / 1_000_000.0,
			max_pulse_width=self.pulse.max_pulse_us / 1_000_000.0,
			initial_angle=physical_initial,
		)
		logging.debug(
			"Servo on pin %d initialised at logical %d° (physical %d°)",
			pin,
			logical_initial,
			physical_initial,
		)
		return ServoConfig(
			servo=servo,
			id=servo_id,
			pin=pin,
			start_angle=start_angle,
			end_angle=end_angle,
			current_angle=logical_initial,
			is_reversed=is_reversed,
		)

	def _apply_angle(self, state: ServoConfig, logical_angle: int) -> None:
		"""論理角度を物理角度に変換してサーボに適用する。is_reversed=Trueの場合は反転する。"""

		physical = logical_angle
		if state.is_reversed:
			physical = self.motion.max_angle - logical_angle
		state.servo.angle = physical

	def _set_led(self, on: bool) -> None:
		"""LEDの状態を設定する。"""

		if self.led is None:
			return

		if on:
			self.led.on()
		else:
			self.led.off()
		logging.debug("LED %s", "ON" if on else "OFF")

	def _move_servos(self, angle1: int, angle2: int, interval_ms: int) -> None:
		"""両方のサーボを指定された角度に動かす．

		Args:
			angle1: 左サーボの目標角度。
			angle2: 右サーボの目標角度。
			interval_ms: サーボの移動間隔（ミリ秒）。
		"""

		left_state = self.servo_states.get("left")
		right_state = self.servo_states.get("right")

		if left_state is None or right_state is None:
			raise RuntimeError("Servos not initialised")

		target1 = clamp(angle1, 0, self.motion.max_angle)
		target2 = clamp(angle2, 0, self.motion.max_angle)

		current1 = left_state.current_angle
		current2 = right_state.current_angle

		step1 = 0 if target1 == current1 else (1 if target1 > current1 else -1)
		step2 = 0 if target2 == current2 else (1 if target2 > current2 else -1)

		steps1 = abs(target1 - current1)
		steps2 = abs(target2 - current2)
		total_steps = max(steps1, steps2)

		logging.debug(
			"Moving servos from (%d°, %d°) to (%d°, %d°) over %d steps",
			current1,
			current2,
			target1,
			target2,
			total_steps,
		)

		try:
			for step in range(total_steps):
				if step < steps1:
					current1 += step1
					left_state.current_angle = current1
					self._apply_angle(left_state, current1)

				if step < steps2:
					current2 += step2
					right_state.current_angle = current2
					self._apply_angle(right_state, current2)

				time.sleep(self.motion.step_delay_s)

			left_state.current_angle = target1
			right_state.current_angle = target2
			self._apply_angle(left_state, target1)
			self._apply_angle(right_state, target2)

			time.sleep(max(interval_ms, 0) / 1000.0)
		finally:
			for servo_state in (left_state, right_state):
				if servo_state is None:
					continue
				try:
					if getattr(servo_state.servo, "pwm_device", None) is not None:
						servo_state.servo.detach()
				except Exception as exc:
					logging.debug("Skipping servo detach: %s", exc)

	def _move_to_start(self) -> None:
		"""2つのサーボを設定された開始角度に移動させる。"""

		self._move_servos(self.motion.start_angle_left, self.motion.start_angle_right, self.motion.rotate_delay_ms)

	def _move_to_end(self) -> None:
		"""2つのサーボを設定された終了角度に移動させる。"""

		self._move_servos(self.motion.end_angle_left, self.motion.end_angle_right, self.motion.rotate_delay_ms)

	def _execute_pour_cycle(self) -> None:
		"""自動注入サイクルを1回実行する。"""

		self._move_to_start()
		self._move_to_end()
		time.sleep(self.motion.pour_time_ms / 1000.0)
		self._move_to_start()

	def _pour_auto(self, is_pressed: bool) -> None:
		"""ボタンがトリガーされたときに自動注入ルーチンを実行する。"""

		if not is_pressed:
			return

		logging.debug("Auto pour triggered")
		time.sleep(self.debounce_delay_s)
		if self.button_pour is not None:
			self.button_pour.wait_for_inactive()

		self._execute_pour_cycle()

	def _timer(self, is_pressed: bool) -> None:
		"""ログにボタン1の押下時間をミリ秒単位で記録する。"""

		if is_pressed:
			if self.press_start_time is None:
				self.press_start_time = time.monotonic()
				logging.debug("Timer start")
		else:
			if self.press_start_time is not None:
				duration_ms = (time.monotonic() - self.press_start_time) * 1000.0
				self.press_start_time = None
				logging.info("Button 1 was pressed for %.0f ms", duration_ms)

	def _handle_manual_mode(
		self,
		button1_pressed: bool,
		button3_pressed: bool,
		button4_pressed: bool,
	) -> None:
		"""手動モードでのサーボ制御を処理する．"""

		left_state = self.servo_states.get("left")
		right_state = self.servo_states.get("right")
		if left_state is None or right_state is None:
			raise RuntimeError("Servos not initialised")

		if button1_pressed:
			# 両サーボをend側へ
			if left_state.current_angle != self.motion.end_angle_left:
				self._ensure_servo_position(left_state, self.motion.end_angle_left)
			if right_state.current_angle != self.motion.end_angle_right:
				self._ensure_servo_position(right_state, self.motion.end_angle_right)
		elif button3_pressed:
			# 左end、右start
			if left_state.current_angle != self.motion.end_angle_left:
				self._ensure_servo_position(left_state, self.motion.end_angle_left)
			if right_state.current_angle != self.motion.start_angle_right:
				self._ensure_servo_position(right_state, self.motion.start_angle_right)
		elif button4_pressed:
			# 左start、右end
			if left_state.current_angle != self.motion.start_angle_left:
				self._ensure_servo_position(left_state, self.motion.start_angle_left)
			if right_state.current_angle != self.motion.end_angle_right:
				self._ensure_servo_position(right_state, self.motion.end_angle_right)
		else:
			# 両サーボをstart側へ
			if left_state.current_angle != self.motion.start_angle_left:
				self._ensure_servo_position(left_state, self.motion.start_angle_left)
			if right_state.current_angle != self.motion.start_angle_right:
				self._ensure_servo_position(right_state, self.motion.start_angle_right)

	def _process_queue_messages(self, apply_commands: bool) -> bool:
		"""キューからの保留中のメッセージを受け取り，必要に応じてサーボを制御する。

		Args:
			apply_commands: True のときはキュー指令を実行し，False のときは指令を破棄する。
		Returns:
			loop()関数を継続するかどうか。

		キューの指令は以下の通り：
		- "start_pour": 両方のサーボで注入を開始
		- "stop_pour": 両方のサーボで注入を停止
		- "start_pour_left": 左サーボで注入を開始
		- "stop_pour_left": 左サーボで注入を停止
		- "start_pour_right": 右サーボで注入を開始
		- "stop_pour_right": 右サーボで注入を停止
		- "start_pour_center": 両方のサーボで注入を開始
		- "stop_pour_center": 両方のサーボで注入を停止
		- "shutdown": サーボ制御ループを終了する。
		"""

		if self.command_queue is None:
			return True

		continue_running = True

		while True:
			try:
				message = self.command_queue.get_nowait()
			except queue.Empty:
				break

			if message is None:
				logging.info("Queue sentinel received; exiting servo loop")
				continue_running = False
				break

			command = message.get("type") if isinstance(message, dict) else message

			if command == "shutdown":
				logging.info("Shutdown command received; exiting servo loop")
				continue_running = False
				break
			elif not apply_commands:
				logging.debug("Ignoring queue message in manual mode: %s", message)
				continue
			elif command == "start_pour":
				self._start_pour_both(message)
			elif command == "stop_pour":
				self._stop_pour_both(message)
			elif command in {"start_pour_left", "start_pour_right"}:
				side = "left" if command.endswith("left") else "right"
				self._start_pour_side(side, message)
			elif command in {"stop_pour_left", "stop_pour_right"}:
				side = "left" if command.endswith("left") else "right"
				self._stop_pour_side(side, message)
			elif command == "start_pour_center":
				self._start_pour_both(message)
			elif command == "stop_pour_center":
				self._stop_pour_both(message)
			elif command == "trigger_pour":
				logging.debug("Legacy trigger command received")
				self._execute_pour_cycle()
			else:
				logging.debug("Ignoring unknown queue message: %s", message)

		return continue_running

	def _cleanup(self) -> None:
		"""サーボ，ボタン，LEDのクリーンアップを行う．"""

		# サーボが初期化済みであれば開始角度へ戻す
		left_state = self.servo_states.get("left")
		right_state = self.servo_states.get("right")
		if left_state is not None and right_state is not None:
			logging.debug("Moving servos to start position for cleanup")
			try:
				self._move_to_start()
			except RuntimeError as exc:
				logging.debug("Skipping servo reset during cleanup: %s", exc)

		logging.debug("Cleaning up gpiozero devices")

		for state in self.servo_states.values():
			state.servo.detach()
			state.servo.close()

		self.servo_states.clear()

		if self.led is not None:
			self.led.off()
			self.led.close()
			self.led = None

		for attribute in ("button_pour", "button_mode", "button_left", "button_right"):
			btn = getattr(self, attribute)
			if btn is not None:
				btn.close()
				setattr(self, attribute, None)

		for side in self.pour_hold_active:
			self.pour_hold_active[side] = False
		self.is_manual = False
		self.press_start_time = None

	def _initialise(self) -> None:
		"""ログの設定、GPIO、サーボ、および初期位置を設定する。"""

		self._setup_logging()
		self._setup_devices()
		self._move_to_start()

	def register_signal_handlers(self) -> None:
		"""Ensure we clean up on termination signals."""

		def _handle_signal(signum, frame):  # pragma: no cover - signal handler
			logging.info("Received signal %s, shutting down", signum)
			self._cleanup()
			sys.exit(0)

		for sig in (signal.SIGINT, signal.SIGTERM):
			signal.signal(sig, _handle_signal)

	def loop(self) -> None:
		"""メインの制御ループ。手動モードと自動モードを処理し、キューからのメッセージを監視する。"""

		assert (
			self.button_pour is not None
			and self.button_mode is not None
			and self.button_left is not None
			and self.button_right is not None
		)

		running = True
		while running:
			# キューからのメッセージを処理する（マニュアルモードの場合は無視する）
			if self.command_queue is not None:
				if not self._process_queue_messages(apply_commands=not self.is_manual):
					running = False
					break

			button_pour_state = self.button_pour.is_active
			button_mode_state = self.button_mode.is_active
			button_left_state = self.button_left.is_active
			button_right_state = self.button_right.is_active

			# マニュアルモードの切り替えを処理する
			if button_mode_state:
				time.sleep(self.debounce_delay_s)
				self.button_mode.wait_for_inactive()
				self.is_manual = not self.is_manual
				self._set_led(self.is_manual)
				logging.info("Manual mode %s", "ENABLED" if self.is_manual else "DISABLED")
				# マニュアルモードが有効で、注入がアクティブな場合は、サーボを初期位置に戻す
				if self.is_manual and any(self.pour_hold_active.values()):
					self._move_to_start()
					for side in self.pour_hold_active:
						self.pour_hold_active[side] = False

			if self.is_manual:
				self._handle_manual_mode(button_pour_state, button_left_state, button_right_state)
			else:
				# 自動モードでは、注入ボタンが押されたときに注入サイクルを実行する
				if not any(self.pour_hold_active.values()):
					self._pour_auto(button_pour_state)

			self._timer(button_pour_state)
			time.sleep(0.01)

	def run(self) -> None:
		"""制御ループをコマンドラインから実行するエントリポイント。もしキューが提供されれば、それを使用してコマンドを受信する。"""

		try:
			self._initialise()
			self.register_signal_handlers()
			if self.command_queue is not None:
				logging.info("Queue is ready.")
				logging.info("Servo control ready. Press Ctrl+C to exit.")
			else:
				logging.info("No command queue provided; running in standalone mode.")
				logging.info("Servo control ready. Press Ctrl+C to exit.")
			self.loop()
		except KeyboardInterrupt:
			logging.info("Interrupted by user. Cleaning up.")
		finally:
			self._cleanup()

	def _start_pour_side(self, side: str, payload) -> None:
		"""キューでの指令を受けて指定されたサーボ側で注入を開始する。
		
		Args:
			side: 'left'または'right'のいずれか。
			payload: コマンドのペイロード（デバッグ用）。
		"""
		if side not in self.servo_states:
			logging.debug("Requested start for unknown side '%s'", side)
			return

		if self.pour_hold_active.get(side, False):
			return

		logging.debug("requested start to pour %s: %s", side, payload)
		left_state = self.servo_states.get("left")
		right_state = self.servo_states.get("right")
		if left_state is None or right_state is None:
			raise RuntimeError("Servos not initialised")

		target_left = left_state.current_angle
		target_right = right_state.current_angle

		if side == "left":
			target_left = self.motion.end_angle_left
		elif side == "right":
			target_right = self.motion.end_angle_right

		self._move_servos(target_left, target_right, self.motion.rotate_delay_ms)
		self.pour_hold_active[side] = True

	def _start_pour_both(self, payload) -> None:
		"""両サーボの注入を同時に開始する補助関数。"""

		left_state = self.servo_states.get("left")
		right_state = self.servo_states.get("right")
		if left_state is None or right_state is None:
			raise RuntimeError("Servos not initialised")

		left_active = self.pour_hold_active.get("left", False)
		right_active = self.pour_hold_active.get("right", False)

		if left_active and right_active:
			logging.debug("Both sides already pouring; ignoring start: %s", payload)
			return

		target_left = left_state.current_angle
		target_right = right_state.current_angle
		if not left_active:
			target_left = self.motion.end_angle_left
		if not right_active:
			target_right = self.motion.end_angle_right

		logging.debug("requested start to pour both: %s", payload)
		self._move_servos(target_left, target_right, self.motion.rotate_delay_ms)
		if not left_active:
			self.pour_hold_active["left"] = True
		if not right_active:
			self.pour_hold_active["right"] = True

	def _stop_pour_side(self, side: str, payload) -> None:
		"""キューでの指令を受けて指定されたサーボ側で注入を停止する。

		Args:
			side: 'left'または'right'のいずれか。
			payload: コマンドのペイロード（デバッグ用）。
		"""
		if side not in self.servo_states:
			logging.debug("Requested stop for unknown side '%s'", side)
			return

		if not self.pour_hold_active.get(side, False):
			return

		logging.debug("requested stop to pour %s: %s", side, payload)
		state = self.servo_states[side]
		target_angle = state.start_angle

		self._ensure_servo_position(state, target_angle, force=True)
		self.pour_hold_active[side] = False

	def _stop_pour_both(self, payload) -> None:
		"""両サーボの注入を同時に停止する補助関数。"""

		left_state = self.servo_states.get("left")
		right_state = self.servo_states.get("right")
		if left_state is None or right_state is None:
			raise RuntimeError("Servos not initialised")

		left_active = self.pour_hold_active.get("left", False)
		right_active = self.pour_hold_active.get("right", False)
		if not left_active and not right_active:
			logging.debug("Both sides already stopped; ignoring stop: %s", payload)
			return

		if left_active:
			self._ensure_servo_position(left_state, left_state.start_angle, force=True)
			self.pour_hold_active["left"] = False

		if right_active:
			self._ensure_servo_position(right_state, right_state.start_angle, force=True)
			self.pour_hold_active["right"] = False

	def _ensure_servo_position(self, state: ServoConfig, target_angle: int, retries: int = 3, settle_s: float = 0.05, *, force: bool = False) -> None:
		"""Try to bring the specified servo to target angle using _move_servos.

		Note:
		- We can't read actual servo feedback, so we avoid comparing physical angles.
		- We use _move_servos so that the "finally" block detaches PWM and prevents jitter.

		Args:
			state: 対象サーボの設定/状態。
			target_angle: 論理角度での目標位置。
			retries: 再送回数（同じコマンドを重ねて送る）。
			settle_s: 各試行後の安定化待ち時間（秒）。
		"""
		# すでに論理的に目的地にいるときは無駄な再送を避ける（force時は送る）
		if not force and state.current_angle == clamp(target_angle, 0, self.motion.max_angle):
			return
		other_id = "right" if state.id == "left" else "left"
		other_state = self.servo_states.get(other_id)

		for attempt in range(1, retries + 1):
			# 片側のみを動かす。もう一方は現在角度を維持する。
			if state.id == "left":
				left_target = clamp(target_angle, 0, self.motion.max_angle)
				right_target = other_state.current_angle if other_state is not None else self.motion.start_angle_right
			else:
				left_state = self.servo_states.get("left")
				left_target = left_state.current_angle if left_state is not None else self.motion.start_angle_left
				right_target = clamp(target_angle, 0, self.motion.max_angle)

			logging.debug("ensure(%s) attempt %d -> %d° (other stays)", state.id, attempt, target_angle)
			self._move_servos(left_target, right_target, self.motion.rotate_delay_ms)
			time.sleep(settle_s)

def run(command_queue: Optional[queue.Queue] = None) -> None:
    """制御ループをコマンドラインから実行するエントリポイント。もしキューが提供されれば、それを使用してコマンドを受信する。"""

    controller = ServoController(command_queue=command_queue)
    controller.run()


if __name__ == "__main__":
	run()
