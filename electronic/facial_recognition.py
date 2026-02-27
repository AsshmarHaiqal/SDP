"""
electronic/facial_recognition.py
Facial verification for PillWheel access control.

Enrol from an image file:
    python -c "from electronic.facial_recognition import enroll_from_image; enroll_from_image('alice', 'alice.jpg')"

Enrol from camera:
    python -c "from electronic.facial_recognition import enroll_face; enroll_face('alice')"

Verify with live display:
    python -c "from electronic.facial_recognition import verify_access_live; print(verify_access_live('alice'))"

Dependencies:
    pip install face_recognition opencv-python numpy
"""

import os
import time

import cv2
import numpy as np

try:
    import face_recognition
    _FR_AVAILABLE = True
except ImportError:
    _FR_AVAILABLE = False

_FACES_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "faces")

# Camera index – run list_cameras() to find the right index on your machine.
CAMERA_INDEX = 1


# ── Enrolment ────────────────────────────────────────────────────────────────

def enroll_face(name: str) -> bool:
    """Capture a face from the camera and save it under the given name."""
    image = _capture_frame()
    if image is None:
        return False
    return _save_encoding(name, image)


def enroll_from_image(name: str, image_path: str) -> bool:
    """
    Load a JPG/PNG and save the face encoding under the given name.
    Place your image file anywhere and pass the path.

    enroll_from_image("alice", "alice.jpg")
    """
    image = face_recognition.load_image_file(image_path)  # returns RGB array
    return _save_encoding(name, image)


# ── Verification ─────────────────────────────────────────────────────────────

def verify_access_live(name: str, override: bool = False) -> bool:
    """
    Open a live camera window.
      - Red box   : face detected, not recognised
      - Green box : face matches enrolled person → closes after 1 s, returns True
    Press 'q' to quit without verifying (returns False).
    """
    if override:
        return True

    if not _FR_AVAILABLE:
        return False

    reference = _load_encoding(name)
    if reference is None:
        return False

    cap = cv2.VideoCapture(CAMERA_INDEX)
    verified = False
    verified_at = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process at half resolution for speed; scale locations back up
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        for (top, right, bottom, left), encoding in zip(locations, encodings):
            top    *= 2; right  *= 2
            bottom *= 2; left   *= 2

            match = face_recognition.compare_faces([reference], encoding)[0]

            if match:
                color = (0, 255, 0)   # green
                label = "Verified"
                if not verified:
                    verified    = True
                    verified_at = time.time()
            else:
                color = (0, 0, 255)   # red
                label = "Unknown"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("PillWheel - Face Verification", frame)

        # Hold green frame for 1 second then close
        if verified and time.time() - verified_at >= 1.0:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return verified


def verify_access(name: str, override: bool = False) -> bool:
    """Silent single-shot verification (no display). Used in production logic."""
    if override:
        return True

    if not _FR_AVAILABLE:
        return False

    image = _capture_frame()
    if image is None:
        return False

    encodings = face_recognition.face_encodings(image)
    person_encoding = encodings[0] if encodings else None

    return facial_verification(person_encoding, _load_encoding(name))


def facial_verification(person_encoding, reference_encoding) -> bool:
    """Compare captured encoding against a stored reference."""
    if person_encoding is None or reference_encoding is None:
        return False
    return bool(face_recognition.compare_faces([reference_encoding], person_encoding)[0])


# ── Utilities ─────────────────────────────────────────────────────────────────

def list_enrolled() -> list:
    """Return names of all enrolled people."""
    if not os.path.exists(_FACES_DIR):
        return []
    return [f.replace(".npy", "") for f in os.listdir(_FACES_DIR) if f.endswith(".npy")]


def list_cameras():
    """Print available camera indices."""
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            print(f"  [{i}] camera available")
        cap.release()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save_encoding(name: str, rgb_image) -> bool:
    encodings = face_recognition.face_encodings(rgb_image)
    if not encodings:
        return False
    os.makedirs(_FACES_DIR, exist_ok=True)
    np.save(_encoding_path(name), encodings[0])
    return True


def _encoding_path(name: str) -> str:
    return os.path.join(_FACES_DIR, f"{name}.npy")


def _load_encoding(name: str):
    path = _encoding_path(name)
    return np.load(path) if os.path.exists(path) else None


def _capture_frame():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
