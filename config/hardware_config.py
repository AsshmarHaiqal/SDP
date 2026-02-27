# Hardware pins (GPIO pins for Raspberry Pi)
IR_SENSOR_PIN = 17
MOTOR_PIN = 27
DISPLAY_PINS = [22, 23, 24]

# Hardware limits
MAX_ROTATES = 5
MOTOR_ROTATION_SPEED = 100  # RPM
MOTOR_STEP_ANGLE = 1.8  # degrees per step

# Sensor thresholds
IR_SENSOR_THRESHOLD = 0.5  # voltage threshold
SENSOR_READ_DELAY = 0.1  # seconds

# Timing
DISPENSE_TIMEOUT = 30  # seconds
ROTATION_DELAY = 0.5  # seconds between rotations

# ── PCA9685 / ServoKit ───────────────────────────────────────────────────────
# The PCA9685 exposes 16 PWM channels (0–15).
# Channels 0–13 are automatically mapped 1-to-1 to dispenser slots 1–14.
# Channels 14–15 are reserved for auxiliary / special-purpose servos.

SERVO_KIT_CHANNELS = 16        # total channels on PCA9685 board
SERVO_FREQUENCY    = 50        # Hz (standard for hobby servos)

DISPENSER_COUNT          = 14                    # number of dispenser slots
DISPENSER_CHANNELS       = list(range(14))       # [0, 1, ..., 13]

SPECIAL_SERVO_CHANNELS   = [14, 15]              # reserved auxiliary channels
SPECIAL_SERVO_LABEL      = {14: "aux_1", 15: "aux_2"}

# Servo pulse-width limits (microseconds) – adjust per servo model if needed
SERVO_MIN_PULSE = 500   # µs
SERVO_MAX_PULSE = 2500  # µs

# Dispense cycle angles
SERVO_HOME_ANGLE    = 0    # resting position (degrees)
SERVO_DISPENSE_ANGLE = 180  # full-swing position (degrees)