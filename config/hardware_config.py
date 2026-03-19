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
ROTATION_DELAY = 0.5   # seconds between rotations

# ── PCA9685 / ServoKit ───────────────────────────────────────────────────────
# The PCA9685 exposes 16 PWM channels (0–15).
# Channels 0–13 are automatically mapped 1-to-1 to dispenser slots 1–13.
# Channels 14–15 are reserved for auxiliary / special-purpose servos.

SERVO_KIT_CHANNELS = 16        # total channels on PCA9685 board
SERVO_FREQUENCY    = 50        # Hz (standard for hobby servos)

DISPENSER_COUNT          = 13                    # number of dispenser slots
DISPENSER_CHANNELS       = list(range(13))       # [0, 1, ..., 12]

SPECIAL_SERVO_CHANNELS   = [14, 15]              # reserved auxiliary channels
SPECIAL_SERVO_LABEL      = {14: "camera_control", 15: "tray_tilt"}

# Servo pulse-width limits (microseconds) – adjust per servo model if needed
SERVO_MIN_PULSE = 500   # µs
SERVO_MAX_PULSE = 2500  # µs

# Dispense cycle angles
SERVO_HOME_ANGLE     = 0    # resting position (degrees) set 3 degree
SERVO_DISPENSE_ANGLE = 180  # full-swing position (degrees) 140

# ── Pill Detection / Dispensing ───────────────────────────────────────────────
# Auxiliary servo slot indices into SPECIAL_SERVO_CHANNELS.
# SPECIAL_SERVO_CHANNELS[0] = PCA9685 channel 14 (camera_control)
# SPECIAL_SERVO_CHANNELS[1] = PCA9685 channel 15 (tray_tilt)
CAMERA_TILT_SERVO_INDEX = 0
TRAY_TILT_SERVO_INDEX   = 1

# Camera repositioning
CAMERA_TILT_ANGLE    = 90    # degrees — points camera downward at tray
CAMERA_RETURN_ANGLE  = 0     # degrees — forward-facing (face recognition position)
CAMERA_SETTLE_DELAY  = 1.0   # seconds — wait after repositioning for image to stabilise

# Dispense verification
PILL_SETTLE_DELAY      = 3.0  # seconds — wait for pill to settle on tray after release
MAX_DISPENSE_ATTEMPTS  = 5    # retries per pill before halting the session

# User confirmation
CONFIRM_TIMEOUT = 10  # seconds — wait for resident confirmation before flagging timeout

# Audit image storage
AUDIT_IMAGE_DIR = "audit_images"  # directory for dispensing audit photos

# ── Claude Vision API ─────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = "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"
CLAUDE_MODEL      = "claude-haiku-4-5-20251001"
