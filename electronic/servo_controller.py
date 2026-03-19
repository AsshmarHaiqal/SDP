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
)


class ServoController:
    """
    Controls all PCA9685 servos.
    Dispenser slot N uses channel N (0-based, channels 0–12).
    Channels 14–15 are auxiliary.
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

    def cleanup(self): #Return all servos to 0°
        if self.hardware_available:
            for ch in range(SERVO_KIT_CHANNELS):
                try:
                    self.kit.servo[ch].angle = 0
                except Exception:
                    pass

    def _rotate(self, channel: int):
        if self.hardware_available:
            self.kit.servo[channel].angle = 0
            time.sleep(ROTATION_DELAY)
            self.kit.servo[channel].angle = 180
            time.sleep(ROTATION_DELAY)
            self.kit.servo[channel].angle = 0
            time.sleep(ROTATION_DELAY)
        else:
            time.sleep(ROTATION_DELAY * 3)
