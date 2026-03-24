"""
electronic/face_tracking.py

Servo-assisted face tracking for PillWheel.

KEY CHANGE vs original:
  - Camera is NO LONGER opened at module-import time.
  - Use open_camera() / close_camera() explicitly, OR use the context manager:

        with camera_open():
            face, frame, angle = scan_for_face(DEFAULT_ANGLE_CAMERA)

  - cap is None until open_camera() succeeds.
  - close_camera() properly releases the V4L2 handle so the next run finds
    /dev/video0 free.  The old version left it locked if Python exited
    uncleanly (abort / SIGKILL), causing the "need to kill it" symptom.
"""

import cv2
import time
import contextlib

try:
    from adafruit_servokit import ServoKit
    _kit = ServoKit(channels=16)
    _kit.servo[14].set_pulse_width_range(400, 2600)
    SERVO_AVAILABLE = True
except Exception as _e:
    _kit = None
    SERVO_AVAILABLE = False
    print(f"face_tracking: ServoKit unavailable ({_e}) — servo calls are no-ops")

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_ANGLE_CAMERA   = 180
MIN_ANGLE              = 0
MAX_ANGLE              = 180
SCAN_STEP              = 5
TRACK_STEP             = 3
FRAME_CENTER_TOLERANCE = 30   # pixels
MIN_FACE_AREA          = 6000  # ~78×78 px — reject tiny/distant faces

CAMERA_DEVICE          = '/dev/video0'
CAMERA_OPEN_RETRIES    = 5
CAMERA_RETRY_DELAY     = 1.0  # seconds between retries

# ── Module-level camera handle (None until open_camera() is called) ───────────
cap = None

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


# ── Camera lifecycle ──────────────────────────────────────────────────────────

def open_camera() -> bool:
    """
    Open /dev/video0 with retries.
    Returns True on success, False if camera could not be opened.
    Sets the module-level `cap`.
    """
    global cap

    # If already open, nothing to do
    if cap is not None and cap.isOpened():
        return True

    # Release any stale handle first
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
        cap = None

    for attempt in range(CAMERA_OPEN_RETRIES):
        _cap = cv2.VideoCapture(CAMERA_DEVICE)
        if _cap.isOpened():
            cap = _cap
            print(f"face_tracking: camera opened ✔ ({CAMERA_DEVICE})")
            return True
        _cap.release()
        print(f"face_tracking: camera not ready, retrying ({attempt+1}/{CAMERA_OPEN_RETRIES})…")
        time.sleep(CAMERA_RETRY_DELAY)

    print(f"face_tracking: ERROR — could not open {CAMERA_DEVICE} after {CAMERA_OPEN_RETRIES} attempts")
    cap = None
    return False


def close_camera() -> None:
    """
    Release the camera handle.  Safe to call even if already closed.
    Always call this on exit so V4L2 frees the device for the next run.
    """
    global cap
    if cap is not None:
        try:
            cap.release()
            print("face_tracking: camera released ✔")
        except Exception as e:
            print(f"face_tracking: warning during release — {e}")
        cap = None


@contextlib.contextmanager
def camera_open():
    """
    Context manager — opens camera on enter, always releases on exit.

    Usage:
        with camera_open():
            face, frame, angle = scan_for_face(DEFAULT_ANGLE_CAMERA)
    """
    opened = open_camera()
    try:
        yield opened
    finally:
        close_camera()


# ── Servo helpers ─────────────────────────────────────────────────────────────

def set_servo_angle(angle: float) -> float:
    angle = max(MIN_ANGLE, min(MAX_ANGLE, float(angle)))
    if SERVO_AVAILABLE:
        _kit.servo[14].angle = angle
    return angle


# ── Face scan ─────────────────────────────────────────────────────────────────

def scan_for_face(current_angle: float):
    """
    Sweep servo from current_angle down to MIN_ANGLE looking for a face.

    Returns:
        (x, y, w, h), frame, angle   on success
        None, None, angle            if no face found
    
    Requires camera to already be open (call open_camera() first, or use
    the camera_open() context manager).
    """
    if cap is None or not cap.isOpened():
        print("scan_for_face: camera not open — call open_camera() first")
        return None, None, current_angle

    angle = current_angle
    while angle >= MIN_ANGLE:
        angle = set_servo_angle(angle - SCAN_STEP)
        time.sleep(0.35)  # let servo settle

        ret, frame = cap.read()
        if not ret:
            print("scan_for_face: cap.read() failed — camera may have dropped")
            # Attempt a single reopen
            if _try_reopen():
                continue
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            continue

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


# ── Internal helpers ──────────────────────────────────────────────────────────

def _try_reopen() -> bool:
    """Try to recover a dropped camera mid-session."""
    global cap
    print("face_tracking: attempting camera reopen…")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
        cap = None
    time.sleep(1.0)
    _cap = cv2.VideoCapture(CAMERA_DEVICE)
    if _cap.isOpened():
        cap = _cap
        print("face_tracking: camera recovered ✔")
        return True
    _cap.release()
    print("face_tracking: camera reopen failed")
    return False


# ── Standalone test ───────────────────────────────────────────────────────────

def track_face():
    """
    Standalone terminal test.
    Opens camera, sweeps for a face, prints result, releases camera.
    """
    print("track_face: opening camera…")
    if not open_camera():
        print("track_face: cannot open camera — aborting")
        return

    try:
        set_servo_angle(DEFAULT_ANGLE_CAMERA)
        time.sleep(0.5)

        face, frame, angle = scan_for_face(DEFAULT_ANGLE_CAMERA)

        if face is None:
            print("track_face: no face found in sweep")
            return

        x, y, w, h = face
        print(f"track_face: face at ({x},{y}) size={w}×{h} servo={angle:.0f}°")

        # Track until centred (terminal mode — no imshow)
        deadline = time.time() + 10
        while cap is not None and cap.isOpened() and time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                break
            fh = frame.shape[0]
            center_y = fh // 2
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                print("track_face: face lost")
                break
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            diff = (y + h // 2) - center_y
            print(f"  servo={angle:.0f}°  diff={diff:+d}px")
            if abs(diff) <= FRAME_CENTER_TOLERANCE:
                print("track_face: face centred ✔")
                break
            angle = set_servo_angle(angle + TRACK_STEP if diff > 0 else angle - TRACK_STEP)
            time.sleep(0.1)

    finally:
        set_servo_angle(DEFAULT_ANGLE_CAMERA)
        close_camera()   # ← always release, even if we crash


if __name__ == "__main__":
    track_face()