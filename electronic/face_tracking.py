import cv2
from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16)
kit.servo[14].set_pulse_width_range(400, 2600)

DEFAULT_ANGLE_CAMERA = 180
MIN_ANGLE = 0
MAX_ANGLE = 180
SCAN_STEP = 5
TRACK_STEP = 3
FRAME_CENTER_TOLERANCE = 30 #pixels

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

def set_servo_angle(angle):
    angle = max (MIN_ANGLE, min(MAX_ANGLE, angle))
    kit.servo[14].angle = angle
    return angle

def scan_for_face(current_angle):
    angle = current_angle
    while angle >= MIN_ANGLE:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            #print(f"Face found at angle {angle}")
            return faces[0], frame, angle
        
        angle = set_servo_angle(angle- SCAN_STEP)
        time.sleep(0.3)

    return None, None, angle

def track_face():
    current_angle = set_servo_angle(DEFAULT_ANGLE_CAMERA)
    time.sleep(0.5)

    face, frame, current_angle = scan_for_face(current_angle)
    
    if face is None:
        set_servo_angle(DEFAULT_ANGLE_CAMERA)
        return None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]
        frame_center_y = frame_height // 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            face, frame, current_angle = scan_for_face(current_angle)
            if face is None:
                break
            continue

        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        face_center_y = y + h // 2

        # Draw debug info on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.line(frame, (0, frame_center_y), (frame_width, frame_center_y), (255, 0, 0), 1)
        cv2.circle(frame, (x + w//2, face_center_y), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Servo: {current_angle:.0f}deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Face Tracker", frame)

        # Check if face is centred
        diff = face_center_y - frame_center_y

        if abs(diff) <= FRAME_CENTER_TOLERANCE:
            captured_frame = frame.copy()
            cv2.imwrite("face_capture.jpg", captured_frame)
            break

        elif diff > FRAME_CENTER_TOLERANCE:
            current_angle = set_servo_angle(current_angle + TRACK_STEP)

        elif diff < -FRAME_CENTER_TOLERANCE:
            current_angle = set_servo_angle(current_angle - TRACK_STEP)

        time.sleep(0.1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    set_servo_angle(DEFAULT_ANGLE_CAMERA)
    cap.release()
    cv2.destroyAllWindows()

track_face()
