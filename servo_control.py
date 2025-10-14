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
from dataclasses import dataclass
from typing import Dict, Optional

from gpiozero import AngularServo, Button, LED  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Configuration constants (mirroring the Arduino sketch)
# ---------------------------------------------------------------------------

DEBUG = True

SERVO_PIN1 = 12  # Left servo (hardware PWM0, channel A)
SERVO_PIN2 = 13  # Right servo (hardware PWM01 channel B)

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
# Module level state
# ---------------------------------------------------------------------------


@dataclass
class ServoConfig:
	"""Keeps track of a servo device and current angle.
	Parameters:
		servo: The gpiozero AngularServo instance.
		id: The unique identifier for the servo.
		pin: The GPIO pin number the servo is attached to.
		start_angle: The angle representing the "start" position.
		end_angle: The angle representing the "end" position.
		current_angle: The last known angle of the servo.
	"""

	servo: AngularServo
	id: str
	pin: int
	start_angle: int
	end_angle: int
	current_angle: int



button1: Optional[Button] = None
button2: Optional[Button] = None
button3: Optional[Button] = None
button4: Optional[Button] = None
led: Optional[LED] = None

is_on: bool = False
press_start_time: Optional[float] = None

# Mantains the live servo objects keyed by their logical identifiers.
servo_states: Dict[str, ServoConfig] = {}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def setup_logging() -> None:
	"""Configure logging based on the debug flag."""

	level = logging.DEBUG if DEBUG else logging.INFO
	logging.basicConfig(
		level=level,
		format="[%(asctime)s] %(levelname)s: %(message)s",
		datefmt="%H:%M:%S",
	)


def setup_devices() -> None:
	"""Initialise gpiozero devices for servos, buttons, and LED."""

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
				current_angle=START_ANGLE1,
			),
			setup_servo(
				id="right",
				pin=SERVO_PIN2,
				start_angle=START_ANGLE2,
				end_angle=END_ANGLE2,
				current_angle=START_ANGLE2,
			),
		)
	}


def angle_to_duty_cycle(angle: int) -> float:
	"""Convert an angle in degrees to the corresponding PWM duty cycle."""

	angle = clamp(angle, 0, MAX_ANGLE)
	pulse_span = SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US
	pulse = SERVO_MIN_PULSE_US + (pulse_span * angle / MAX_ANGLE)
	period_us = 1_000_000.0 / PWM_FREQUENCY
	duty_cycle = (pulse / period_us) * 100.0
	return duty_cycle


def clamp(value: int, lower: int, upper: int) -> int:
	"""Clamp *value* to the inclusive range [lower, upper]."""

	return max(lower, min(upper, value))


def setup_servo(
		id: str,
		pin: int,
		start_angle: int,
		end_angle: int,
		current_angle: int) -> ServoConfig:
	"""Create and start an angular servo at *start_angle*."""

	servo = AngularServo(
		pin,
		min_angle=0,
		max_angle=MAX_ANGLE,
		min_pulse_width=SERVO_MIN_PULSE_US / 1_000_000.0,
		max_pulse_width=SERVO_MAX_PULSE_US / 1_000_000.0,
		initial_angle=clamp(start_angle, 0, MAX_ANGLE),
	)
	logging.debug("Servo on pin %d initialised at %d°", pin, start_angle)
	return ServoConfig(
		servo=servo, id=id, pin=pin, start_angle=start_angle, end_angle=end_angle, current_angle=current_angle
	)


def cleanup() -> None:
	"""Stop servo outputs and release gpiozero resources."""

	logging.debug("Cleaning up gpiozero devices")

	global button1, button2, button3, button4, led, servo_states

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


def set_led(on: bool) -> None:
	"""Set the LED to the desired state."""

	if led is None:
		return

	if on:
		led.on()
	else:
		led.off()
	logging.debug("LED %s", "ON" if on else "OFF")


def is_button_pressed(btn: Optional[Button]) -> bool:
	"""Return True if the button is currently pressed."""

	return bool(btn and btn.is_active)


def wait_for_button_release(btn: Optional[Button]) -> None:
	"""Block until the button transitions to the released state."""

	if btn is None:
		return

	btn.wait_for_inactive()


def move_servos(angle1: int, angle2: int, interval_ms: int) -> None:
	"""Move both servos smoothly to the supplied target angles."""

	left_state = servo_states.get("left")
	right_state = servo_states.get("right")

	if left_state is None or right_state is None:
		raise RuntimeError("Servos not initialised")

	target1 = clamp(angle1, 0, MAX_ANGLE)
	target2 = clamp(angle2, 0, MAX_ANGLE)

	current1 = left_state.current_angle
	current2 = right_state.current_angle

	# Determine the step direction and number of steps needed for each servo.
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
				left_state.servo.angle = current1

			if step < steps2:
				current2 += step2
				right_state.current_angle = current2
				right_state.servo.angle = current2

			time.sleep(STEP_DELAY_S)

		# Ensure we finish exactly on the targets.
		left_state.current_angle = target1
		right_state.current_angle = target2
		left_state.servo.angle = target1
		right_state.servo.angle = target2

		time.sleep(max(interval_ms, 0) / 1000.0)
	finally:
		left_state.servo.detach()
		right_state.servo.detach()


def pour_auto(button1_pressed: bool) -> None:
	"""Execute the automatic pouring routine when button 1 is triggered."""

	if button1_pressed:
		logging.debug("Auto pour triggered")
		time.sleep(DEBOUNCE_DELAY_S)
		wait_for_button_release(button1)

		move_servos(START_ANGLE1, START_ANGLE2, ROTATE_DELAY_MS)
		move_servos(END_ANGLE1, END_ANGLE2, ROTATE_DELAY_MS)
		time.sleep(POURING_TIME_MS / 1000.0)
		move_servos(START_ANGLE1, START_ANGLE2, ROTATE_DELAY_MS)


def timer(button1_pressed: bool) -> None:
	"""Measure and log the duration that button 1 stays pressed."""

	global press_start_time

	if button1_pressed:
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
	"""Drive servos according to manual-mode button inputs."""

	if button1_pressed:
		move_servos(END_ANGLE1, END_ANGLE2, ROTATE_DELAY_MS)
	elif button3_pressed:
		move_servos(END_ANGLE1, START_ANGLE2, ROTATE_DELAY_MS)
	elif button4_pressed:
		move_servos(START_ANGLE1, END_ANGLE2, ROTATE_DELAY_MS)
	else:
		move_servos(START_ANGLE1, START_ANGLE2, ROTATE_DELAY_MS)


def loop() -> None:
	"""Continuous control loop analogous to the Arduino ``loop`` function."""

	global is_on

	assert button1 is not None and button2 is not None and button3 is not None and button4 is not None

	while True:
		button1_state = is_button_pressed(button1)
		button2_state = is_button_pressed(button2)
		button3_state = is_button_pressed(button3)
		button4_state = is_button_pressed(button4)

		if button2_state:
			time.sleep(DEBOUNCE_DELAY_S)
			wait_for_button_release(button2)
			is_on = not is_on
			set_led(is_on)
			logging.info("Manual mode %s", "ENABLED" if is_on else "DISABLED")

		if is_on:
			handle_manual_mode(button1_state, button3_state, button4_state)
		else:
			pour_auto(button1_state)

		timer(button1_state)
		time.sleep(0.01)


def initialise() -> None:
	"""Set up logging, GPIO, servos, and initial positions."""

	setup_logging()
	setup_devices()

	move_servos(START_ANGLE1, START_ANGLE2, ROTATE_DELAY_MS)


def register_signal_handlers() -> None:
	"""Ensure we clean up on termination signals."""

	def _handle_signal(signum, frame):  # pragma: no cover - signal handler
		logging.info("Received signal %s, shutting down", signum)
		cleanup()
		sys.exit(0)

	for sig in (signal.SIGINT, signal.SIGTERM):
		signal.signal(sig, _handle_signal)


def main() -> None:
	"""Entrypoint for running the control loop from the command line."""

	try:
		initialise()
		register_signal_handlers()
		logging.info("Servo control ready. Press Ctrl+C to exit.")
		loop()
	except KeyboardInterrupt:
		logging.info("Interrupted by user. Cleaning up.")
	finally:
		cleanup()


if __name__ == "__main__":
	main()
