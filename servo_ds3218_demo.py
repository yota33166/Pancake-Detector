"""Minimal gpiozero demo for driving a DS3218 servo on BCM GPIO 13.

This script uses the pigpio backend via gpiozero's PiGPIOFactory so it must be
executed on a Raspberry Pi with the pigpiod daemon running::

    sudo systemctl start pigpiod  # or: sudo pigpiod

Once running, the script will sweep the servo across its full travel and then
hold a neutral position. Press Ctrl+C to exit cleanly.
"""

from __future__ import annotations

import signal
import sys
import time

from gpiozero import AngularServo
from gpiozero.pins.lgpio import LGPIOFactory
# Hardware configuration -----------------------------------------------------
PWM_PIN = 13  # BCM numbering
MIN_PULSE_WIDTH_S = 500 / 1_000_000  # 500 µs
MAX_PULSE_WIDTH_S = 2_500 / 1_000_000  # 2500 µs
SWEEP_STEP_DEGREES = 5
SWEEP_DELAY_S = 0.05
NEUTRAL_ANGLE = 90


def build_servo() -> AngularServo:
    """Create an AngularServo configured for the DS3218 using pigpio."""

    factory = LGPIOFactory()
    return AngularServo(
        PWM_PIN,
        min_angle=0,
        max_angle=180,
        min_pulse_width=MIN_PULSE_WIDTH_S,
        max_pulse_width=MAX_PULSE_WIDTH_S,
        initial_angle=NEUTRAL_ANGLE,
        pin_factory=factory,
    )


def sweep_once(servo: AngularServo) -> None:
    """Sweep the servo from 0° to 180° and back."""

    for angle in range(0, 181, SWEEP_STEP_DEGREES):
        servo.angle = angle
        time.sleep(SWEEP_DELAY_S)

    for angle in range(180, -1, -SWEEP_STEP_DEGREES):
        servo.angle = angle
        time.sleep(SWEEP_DELAY_S)


def main() -> None:
    servo = build_servo()

    def _handle_signal(signum, frame):  # pragma: no cover - signal handler
        servo.detach()
        servo.close()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    try:
        while True:
            sweep_once(servo)
            servo.angle = NEUTRAL_ANGLE
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        servo.detach()
        servo.close()


if __name__ == "__main__":
    main()
