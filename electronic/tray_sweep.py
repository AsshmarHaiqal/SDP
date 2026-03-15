"""
Tray sweep module for the PillWheel dispenser.

Drives the tray servo on PCA9685 channel 14 through a configurable number of
full 0 → 180 → 0 degree sweeps to settle pills into the collection cup.
"""

import time

try:
    from adafruit_servokit import ServoKit
    _SERVOKIT_AVAILABLE = True
except ImportError:
    _SERVOKIT_AVAILABLE = False

from config.hardware_config import SERVO_KIT_CHANNELS, ROTATION_DELAY, SPECIAL_SERVO_CHANNELS

TRAY_CHANNEL = SPECIAL_SERVO_CHANNELS[1]   # PCA9685 channel 14
SWEEP_CYCLES = 2


def sweep(cycles=SWEEP_CYCLES):
    """
    Sweep the tray servo from 0 → 180 → 0 degrees, repeated `cycles` times.
    Uses set_pulse_width_range(400, 2600) calibrated for this servo.
    """
    if not _SERVOKIT_AVAILABLE:
        print(f"[tray_sweep] Hardware not available — simulating {cycles} sweep(s).")
        time.sleep(ROTATION_DELAY * 2 * cycles)
        return

    kit = ServoKit(channels=SERVO_KIT_CHANNELS)
    kit.servo[TRAY_CHANNEL].set_pulse_width_range(400, 2600)

    for _ in range(cycles):
        kit.servo[TRAY_CHANNEL].angle = 0
        time.sleep(ROTATION_DELAY)
        kit.servo[TRAY_CHANNEL].angle = 180
        time.sleep(ROTATION_DELAY)
        kit.servo[TRAY_CHANNEL].angle = 0
        time.sleep(ROTATION_DELAY)


if __name__ == "__main__":
    print(f"Running {SWEEP_CYCLES} sweep(s) on channel {TRAY_CHANNEL}...")
    sweep()
    print("Done.")
