"""
Model 1: Local face_recognition (dlib-based).
Fast, private, no internet required.
Best for Pi 3.
"""
import os
import time

import cv2
import numpy as np

from .base import FaceRecognitionBase

try:
    import face_recognition
    _FR_AVAILABLE = True
except ImportError:
    _FR_AVAILABLE = False
    print("WARNING: face_recognition not installed")


class LocalFaceRecognition(FaceRecognitionBase):

    def __init__(self, faces_dir: str, camera_index: str = '/dev/video0'):
        self.faces_dir     = faces_dir
        self.camera_index  = camera_index
        os.makedirs(faces_dir, exist_ok=True)

    # ── Enrollment ────────────────────────────────────────────────────

    def enroll(self, name: str, image_path: str = None) -> bool:
        if not _FR_AVAILABLE:
            print(f"Enroll skipped for {name}: face_recognition library not installed")
            return False
        if image_path:
            image = face_recognition.load_image_file(image_path)
        else:
            image = self._capture_frame()
        if image is None:
            return False
        return self._save_encoding(name, image)

    # ── Verification ──────────────────────────────────────────────────

    def verify(self, name: str, override: bool = False) -> bool:
        if override:
            return True
        if not _FR_AVAILABLE:
            return False

        image = self._capture_frame()
        if image is None:
            return False

        encodings = face_recognition.face_encodings(image)
        person_enc = encodings[0] if encodings else None
        return self._compare(person_enc, self._load_encoding(name))

    def verify_live(self, name: str, override: bool = False) -> bool:
        if override:
            return True
        if not _FR_AVAILABLE:
            return False

        reference = self._load_encoding(name)
        if reference is None:
            print(f"No enrollment found for: {name}")
            return False

        cap          = cv2.VideoCapture(self.camera_index)
        verified     = False
        verified_at  = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locations = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, locations)

            for (top, right, bottom, left), encoding in zip(locations, encodings):
                top *= 2; right *= 2; bottom *= 2; left *= 2

                match = face_recognition.compare_faces([reference], encoding)[0]

                if match:
                    colour, label = (0, 255, 0), "Verified"
                    if not verified:
                        verified    = True
                        verified_at = time.time()
                else:
                    colour, label = (0, 0, 255), "Unknown"

                cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
                cv2.putText(frame, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

            cv2.imshow("PillWheel - Verification", frame)

            if verified and time.time() - verified_at >= 1.0:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return verified

    def list_enrolled(self) -> list:
        if not os.path.exists(self.faces_dir):
            return []
        return [f.replace(".npy", "") for f in os.listdir(self.faces_dir)
                if f.endswith(".npy")]

    # ── Internal ──────────────────────────────────────────────────────

    def _save_encoding(self, name: str, rgb_image) -> bool:
        if not _FR_AVAILABLE:
            return False
        encodings = face_recognition.face_encodings(rgb_image)
        if not encodings:
            print(f"No face detected in image for {name}")
            return False
        np.save(self._encoding_path(name), encodings[0])
        print(f"Enrolled: {name}")
        return True

    def _load_encoding(self, name: str):
        path = self._encoding_path(name)
        return np.load(path) if os.path.exists(path) else None

    def _encoding_path(self, name: str) -> str:
        return os.path.join(self.faces_dir, f"{name}.npy")

    def _capture_frame(self):
        cap = cv2.VideoCapture(self.camera_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _compare(self, person_enc, reference_enc) -> bool:
        if person_enc is None or reference_enc is None:
            return False
        return bool(face_recognition.compare_faces([reference_enc], person_enc)[0])