"""
Channel allocation (PCA9685, 16 channels):
    0–12  : Dispenser servos (slot index = channel)
    14–15 : Auxiliary servos for camera and etc
"""

import time

try:
    from adafruit_servokit import ServoKit
    _SERVOKIT_AVAILABLE = True
except ImportError:
    _SERVOKIT_AVAILABLE = False

from config.hardware_config import (
    SERVO_KIT_CHANNELS,
    DISPENSER_CHANNELS,
    SPECIAL_SERVO_CHANNELS,
    SERVO_MIN_PULSE,
    SERVO_MAX_PULSE,
    ROTATION_DELAY,
    SERVO_DEFAULT_ANGLE,
    SERVO_DISPENSE_ANGLE,
    CAMERA_DEFAULT_ANGLE,
)

# Channel numbers for the two special servos
_CH_CAMERA = SPECIAL_SERVO_CHANNELS[0]   # 14 — camera_control
_CH_TRAY   = SPECIAL_SERVO_CHANNELS[1]   # 15 — tray_tilt  (excluded from reset)


class ServoController:
    """
    Controls all PCA9685 servos.
    Dispenser slot N uses channel N (0-based, channels 0–12).
    Channel 13 is empty (reserved).
    Channel 14 = camera_control  — rests at CAMERA_DEFAULT_ANGLE (180°).
    Channel 15 = tray_tilt       — managed by tray_sweep.py only.
    """

    def __init__(self):
        self.kit = None
        self.hardware_available = False

        if _SERVOKIT_AVAILABLE:
            try:
                self.kit = ServoKit(channels=SERVO_KIT_CHANNELS)
                for ch in range(SERVO_KIT_CHANNELS):
                    self.kit.servo[ch].set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)
                self.hardware_available = True
                # Camera servo defaults to CAMERA_DEFAULT_ANGLE — faces down at tray
                self.kit.servo[_CH_CAMERA].angle = CAMERA_DEFAULT_ANGLE
            except Exception:
                pass

    def rotate_dispenser(self, dispenser_index: int):
        self._rotate(DISPENSER_CHANNELS[dispenser_index])

    def rotate_special(self, special_index: int):
        self._rotate(SPECIAL_SERVO_CHANNELS[special_index])

    def set_servo_angle(self, channel: int, angle: float):
        """Set a servo to a specific angle and hold (no sweep pattern)."""
        if self.hardware_available:
            self.kit.servo[channel].angle = float(angle)
            time.sleep(ROTATION_DELAY)
        else:
            time.sleep(ROTATION_DELAY)

    def cleanup(self):
        """
        Reset servos one channel at a time to limit inrush current.

        Sequence:
            ch 0 → 14  : set to SERVO_DEFAULT_ANGLE (0°) sequentially
            ch 14       : correct to CAMERA_DEFAULT_ANGLE (180°) after the loop
            ch 15       : excluded — tray_tilt is managed by tray_sweep.py

        Channel 13 is physically empty but is still cycled through the loop
        harmlessly to keep the code simple and loop-range clean.
        """
        if not self.hardware_available:
            return

        # Sequential reset — channels 0 through 14 inclusive
        for ch in range(0, _CH_TRAY):   # range(0, 15) → 0, 1, …, 14
            try:
                self.kit.servo[ch].angle = SERVO_DEFAULT_ANGLE
                time.sleep(ROTATION_DELAY)
            except Exception:
                pass

        # Restore camera servo to its default down-facing position
        try:
            self.kit.servo[_CH_CAMERA].angle = CAMERA_DEFAULT_ANGLE
        except Exception:
            pass

    def _rotate(self, channel: int):
        if self.hardware_available:
            self.kit.servo[channel].angle = SERVO_DEFAULT_ANGLE
            time.sleep(ROTATION_DELAY)
            self.kit.servo[channel].angle = SERVO_DISPENSE_ANGLE
            time.sleep(ROTATION_DELAY)
            self.kit.servo[channel].angle = SERVO_DEFAULT_ANGLE
            time.sleep(ROTATION_DELAY)
        else:
            time.sleep(ROTATION_DELAY * 3)