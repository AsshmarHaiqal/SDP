#Hardware pins (GPIO pins for Raspberry Pi/Arduino)
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

# API endpoints (for Java communication)
JAVA_API_URL = "http://localhost:8080/api"