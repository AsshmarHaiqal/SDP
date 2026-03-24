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
FRAME_CENTER_TOLERANCE = 30  # pixels
MIN_FACE_AREA = 6000          # ~78x78px — reject faces that are too far/small

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = None
for _attempt in range(5):
    cap = cv2.VideoCapture('/dev/video0')
    if cap.isOpened():
        break
    print(f"face_tracking: camera not ready, retrying ({_attempt+1}/5)…")
    time.sleep(1)

if not cap.isOpened():
    raise RuntimeError("Camera failed to open at /dev/video0")

def set_servo_angle(angle):
    angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
    kit.servo[14].angle = angle
    return angle


def scan_for_face(current_angle):
    angle = current_angle
    while angle >= MIN_ANGLE:
        angle = set_servo_angle(angle - SCAN_STEP)
        time.sleep(0.35)  # let servo settle before reading

        ret, frame = cap.read()
        if not ret:
            print("scan_for_face: cap.read() failed")
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            continue

        # Pick largest face and reject if too small (person too far)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        if w * h < MIN_FACE_AREA:
            continue

        # Sharpness check — reject blurry frames
        face_roi = gray[y:y+h, x:x+w]
        if cv2.Laplacian(face_roi, cv2.CV_64F).var() < 40:
            continue

        print(f"scan_for_face: face found at angle={angle:.0f}°")
        return (x, y, w, h), frame, angle

    return None, None, angle


def track_face():
    """
    Standalone test entry point.
    Prints servo angle and diff to terminal instead of using cv2.imshow
    (which requires Qt fonts not available on this Pi build).
    """
    current_angle = set_servo_angle(DEFAULT_ANGLE_CAMERA)
    time.sleep(0.5)

    print("track_face: scanning for face…")
    face, frame, current_angle = scan_for_face(current_angle)

    if face is None:
        print("track_face: no face found during sweep")
        set_servo_angle(DEFAULT_ANGLE_CAMERA)
        cap.release()
        return None

    print("track_face: face detected — centering…")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("track_face: cap.read() failed")
            break

        frame_height, frame_width = frame.shape[:2]
        frame_center_y = frame_height // 2

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            print("track_face: lost face — re-scanning…")
            face, frame, current_angle = scan_for_face(current_angle)
            if face is None:
                print("track_face: face lost permanently")
                break
            continue

        face          = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h    = face
        face_center_y = y + h // 2
        diff          = face_center_y - frame_center_y

        print(f"  diff={diff:+d}px  servo={current_angle:.0f}°")

        # Check if face is centred
        if abs(diff) <= FRAME_CENTER_TOLERANCE:
            print("track_face: face centred ✔  saving face_capture.jpg")
            cv2.imwrite("face_capture.jpg", frame)
            break

        elif diff > FRAME_CENTER_TOLERANCE:
            current_angle = set_servo_angle(current_angle + TRACK_STEP)
        else:
            current_angle = set_servo_angle(current_angle - TRACK_STEP)

        time.sleep(0.1)

    set_servo_angle(DEFAULT_ANGLE_CAMERA)
    cap.release()
    print("track_face: done")


if __name__ == "__main__":
    track_face()