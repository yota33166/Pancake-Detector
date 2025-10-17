"""Servo control logic for the pancake dispenser running on Raspberry Pi.

This module mirrors the behavior of the supplied Arduino sketch by:

* Driving two hobby servos attached to GPIO pins 5 and 6
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

from gpiozero import AngularServo, Button, LED  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 定数の定義
# ---------------------------------------------------------------------------

DEBUG = True

SERVO_PIN1 = 12  # 左サーボ (hardware PWM0)
SERVO_PIN2 = 13  # 右サーボ (hardware PWM1)

BUTTON_PIN1 = 17
BUTTON_PIN2 = 27
BUTTON_PIN3 = 22
BUTTON_PIN4 = 23

# LED_PIN = 13  # On-board LED (BCM numbering)
led = None

ROTATE_DELAY_MS = 20
POURING_TIME_MS = 1000
MAX_ANGLE = 180
START_ANGLE1 = 20
END_ANGLE1 = 80
START_ANGLE2 = MAX_ANGLE - 20
END_ANGLE2 = MAX_ANGLE - 80

PWM_FREQUENCY = 50  # Hertz (spec range 50-330 Hz)
SERVO_MIN_PULSE_US = 500
SERVO_MAX_PULSE_US = 2500

DEBOUNCE_DELAY_S = 0.01
STEP_DELAY_S = 0.005


# ---------------------------------------------------------------------------
# モジュールレベルの状態
# ---------------------------------------------------------------------------


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



button1: Optional[Button] = None
button2: Optional[Button] = None
button3: Optional[Button] = None
button4: Optional[Button] = None
led: Optional[LED] = None

is_on: bool = False
press_start_time: Optional[float] = None

# キー付けされたサーボオブジェクトを保持
servo_states: Dict[str, ServoConfig] = {}
detector_hold_active: bool = False


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------


def setup_logging() -> None:
	"""ログの設定を行う"""

	level = logging.DEBUG if DEBUG else logging.INFO
	logging.basicConfig(
		level=level,
		format="[%(asctime)s] %(levelname)s: %(message)s",
		datefmt="%H:%M:%S",
	)


def setup_devices() -> None:
	"""サーボ，ボタン，LEDの初期化を行う．"""

	global button1, button2, button3, button4, led, servo_states

	button1 = Button(BUTTON_PIN1, pull_up=True)
	button2 = Button(BUTTON_PIN2, pull_up=True)
	button3 = Button(BUTTON_PIN3, pull_up=True)
	button4 = Button(BUTTON_PIN4, pull_up=True)

	# led = LED(LED_PIN)
	# led.off()


	# IDをキーとしてServoConfigの辞書を作成
	servo_states = {
		cfg.id: cfg
		for cfg in (
			setup_servo(
				id="left",
				pin=SERVO_PIN1,
				start_angle=START_ANGLE1,
				end_angle=END_ANGLE1,
				is_reversed=False,
			),
			setup_servo(
				id="right",
				pin=SERVO_PIN2,
				start_angle=START_ANGLE2,
				end_angle=END_ANGLE2,
				is_reversed=False,
			),
		)
	}


def _apply_angle(state: ServoConfig, logical_angle: int) -> None:
	"""論理角度を物理角度に変換してサーボに適用する。is_reversed=Trueの場合は反転する。"""
	physical = logical_angle
	if state.is_reversed:
		physical = MAX_ANGLE - logical_angle
	state.servo.angle = physical


# def angle_to_duty_cycle(angle: int) -> float:
# 	"""degree単位の角度を対応するPWMデューティサイクルに変換する。"""

# 	angle = clamp(angle, 0, MAX_ANGLE)
# 	pulse_span = SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US
# 	pulse = SERVO_MIN_PULSE_US + (pulse_span * angle / MAX_ANGLE)
# 	period_us = 1_000_000.0 / PWM_FREQUENCY
# 	duty_cycle = (pulse / period_us) * 100.0
# 	return duty_cycle


def clamp(value: int, lower: int, upper: int) -> int:
	"""最小値と最大値の範囲にvalueをクランプする。"""

	return max(lower, min(upper, value))


def setup_servo(
		id: str,
		pin: int,
		start_angle: int,
		end_angle: int,
		is_reversed: bool = True
		) -> ServoConfig:
	"""初期化されたサーボオブジェクトを作成し、ServoConfigを返す。"""

	initial_angle = clamp(start_angle, 0, MAX_ANGLE)
	if is_reversed:
		initial_angle = MAX_ANGLE - initial_angle
	servo = AngularServo(
		pin,
		min_angle=0,
		max_angle=MAX_ANGLE,
		min_pulse_width=SERVO_MIN_PULSE_US / 1_000_000.0,
		max_pulse_width=SERVO_MAX_PULSE_US / 1_000_000.0,
		initial_angle=initial_angle,
	)
	
	logging.debug("Servo on pin %d initialised at %d°", pin, initial_angle)
	return ServoConfig(
		servo=servo, id=id, pin=pin, start_angle=start_angle, end_angle=end_angle, current_angle=initial_angle, is_reversed=is_reversed
	)


def cleanup() -> None:
	"""サーボ，ボタン，LEDのクリーンアップを行う．"""

	logging.debug("Cleaning up gpiozero devices")

	global button1, button2, button3, button4, led, servo_states, detector_hold_active

	for state in servo_states.values():
		state.servo.detach()
		state.servo.close()

	servo_states.clear()

	if led is not None:
		led.off()
		led.close()
		led = None

	for btn_name in ("button1", "button2", "button3", "button4"):
		btn = globals().get(btn_name)
		if btn is not None:
			btn.close()
			globals()[btn_name] = None

	detector_hold_active = False


def set_led(on: bool) -> None:
	"""LEDの状態を設定する。"""

	if led is None:
		return

	if on:
		led.on()
	else:
		led.off()
	logging.debug("LED %s", "ON" if on else "OFF")


# def is_button_pressed(btn: Optional[Button]) -> bool:
# 	"""ボタンが現在押されているかどうかを返す。"""

# 	return bool(btn and btn.is_active)


# def wait_for_button_release(btn: Optional[Button]) -> None:
# 	"""Block until the button transitions to the released state."""

# 	if btn is None:
# 		return

# 	btn.wait_for_inactive()


def move_servos(angle1: int, angle2: int, interval_ms: int) -> None:
	"""両方のサーボを指定された角度に動かす．"""

	left_state = servo_states.get("left")
	right_state = servo_states.get("right")

	if left_state is None or right_state is None:
		raise RuntimeError("Servos not initialised")

	target1 = clamp(angle1, 0, MAX_ANGLE)
	target2 = clamp(angle2, 0, MAX_ANGLE)

	current1 = left_state.current_angle
	current2 = right_state.current_angle

	# 各サーボの移動方向を決定する
	step1 = 0 if target1 == current1 else (1 if target1 > current1 else -1)
	step2 = 0 if target2 == current2 else (1 if target2 > current2 else -1)

	# 移動する角度の絶対値を計算する
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
				_apply_angle(left_state, current1)

			if step < steps2:
				current2 += step2
				right_state.current_angle = current2
				_apply_angle(right_state, current2)

			time.sleep(STEP_DELAY_S)

		# 最終的な目標角度を設定する
		left_state.current_angle = target1
		right_state.current_angle = target2
		_apply_angle(left_state, target1)
		_apply_angle(right_state, target2)

		time.sleep(max(interval_ms, 0) / 1000.0)
	finally:
		left_state.servo.detach()
		right_state.servo.detach()


def move_to_start() -> None:
	"""2つのサーボを設定された開始角度に移動させる。"""

	move_servos(START_ANGLE1, START_ANGLE2, ROTATE_DELAY_MS)


def move_to_end() -> None:
	"""2つのサーボを設定された終了角度に移動させる。"""

	move_servos(END_ANGLE1, END_ANGLE2, ROTATE_DELAY_MS)


def execute_pour_cycle() -> None:
	"""自動注入サイクルを1回実行する。"""

	move_to_start()
	move_to_end()
	time.sleep(POURING_TIME_MS / 1000.0)
	move_to_start()


def pour_auto(is_pressed: bool) -> None:
	"""ボタンがトリガーされたときに自動注入ルーチンを実行する。"""

	if is_pressed:
		logging.debug("Auto pour triggered")
		time.sleep(DEBOUNCE_DELAY_S)
		button1.wait_for_inactive()

		execute_pour_cycle()


def timer(is_pressed: bool) -> None:
	"""ログにボタン1の押下時間をミリ秒単位で記録する。"""

	global press_start_time

	if is_pressed:
		if press_start_time is None:
			press_start_time = time.monotonic()
			logging.debug("Timer start")
	else:
		if press_start_time is not None:
			duration_ms = (time.monotonic() - press_start_time) * 1000.0
			press_start_time = None
			logging.info("Button 1 was pressed for %.0f ms", duration_ms)


def handle_manual_mode(
	button1_pressed: bool,
	button3_pressed: bool,
	button4_pressed: bool,
) -> None:
	"""手動モードでのサーボ制御を処理する．"""

	if button1_pressed:
		move_to_end()
	elif button3_pressed:
		move_servos(END_ANGLE1, START_ANGLE2, ROTATE_DELAY_MS)
	elif button4_pressed:
		move_servos(START_ANGLE1, END_ANGLE2, ROTATE_DELAY_MS)
	else:
		move_to_start()

def process_queue_messages(command_queue: queue.Queue) -> bool:
	"""キューからの保留中のメッセージを受け取り，それに応じてサーボを制御する。"""

	global detector_hold_active

	continue_running = True

	while True:
		try:
			message = command_queue.get_nowait()
		except queue.Empty:
			break

		if message is None:
			logging.info("Queue sentinel received; exiting servo loop")
			continue_running = False
			break

		command = message.get("type") if isinstance(message, dict) else message

		if command == "start_pour":
			if not detector_hold_active:
				logging.debug("Detector requested start: %s", message)
				move_to_end()
				detector_hold_active = True
		elif command == "stop_pour":
			if detector_hold_active:
				logging.debug("Detector requested stop: %s", message)
				move_to_start()
				detector_hold_active = False
		elif command == "trigger_pour":
			logging.debug("Legacy trigger command received")
			execute_pour_cycle()
		elif command == "shutdown":
			logging.info("Shutdown command received; exiting servo loop")
			continue_running = False
			break
		else:
			logging.debug("Ignoring unknown queue message: %s", message)

	return continue_running


def loop(command_queue: Optional[queue.Queue] = None) -> None:
	"""メインの制御ループ。手動モードと自動モードを処理し、キューからのメッセージを監視する。"""

	global is_on, detector_hold_active

	assert button1 is not None and button2 is not None and button3 is not None and button4 is not None

	running = True
	while running:
		if command_queue is not None:
			running = process_queue_messages(command_queue)
			if not running:
				break

		button1_state = button1.is_active
		button2_state = button2.is_active
		button3_state = button3.is_active
		button4_state = button4.is_active

		if button2_state:
			time.sleep(DEBOUNCE_DELAY_S)
			button2.wait_for_inactive()
			is_on = not is_on
			set_led(is_on)
			logging.info("Manual mode %s", "ENABLED" if is_on else "DISABLED")
			if is_on and detector_hold_active:
				move_to_start()
				detector_hold_active = False

		if is_on:
			handle_manual_mode(button1_state, button3_state, button4_state)
		else:
			if not detector_hold_active:
				pour_auto(button1_state)

		timer(button1_state)
		time.sleep(0.01)


def initialise() -> None:
	"""ログの設定、GPIO、サーボ、および初期位置を設定する。"""

	setup_logging()
	setup_devices()

	move_to_start()



def register_signal_handlers() -> None:
	"""Ensure we clean up on termination signals."""

	def _handle_signal(signum, frame):  # pragma: no cover - signal handler
		logging.info("Received signal %s, shutting down", signum)
		cleanup()
		sys.exit(0)

	for sig in (signal.SIGINT, signal.SIGTERM):
		signal.signal(sig, _handle_signal)


def run(command_queue: Optional[queue.Queue] = None) -> None:
	"""制御ループをコマンドラインから実行するエントリポイント。もしキューが提供されれば、それを使用してコマンドを受信する。"""

	try:
		initialise()
		register_signal_handlers()
		if command_queue is not None:
			logging.info("Queue is ready.")
			logging.info("Servo control ready. Press Ctrl+C to exit.")
			loop(command_queue)
		else:
			logging.info("No command queue provided; running in standalone mode.")
			logging.info("Servo control ready. Press Ctrl+C to exit.")
			loop()
	except KeyboardInterrupt:
		logging.info("Interrupted by user. Cleaning up.")
	finally:
		cleanup()


if __name__ == "__main__":
	run()
