"""
electronic/facial_recognition.py
PillWheel facial recognition — switch models via FR_MODEL constant.

Usage:
    from electronic.facial_recognition import FacialRecognition

    fr = FacialRecognition()
    fr.enroll("alice", "alice.jpg")
    fr.verify_live("alice")
"""

import os
from electronic.fr_models import LocalFaceRecognition, ClaudeFaceRecognition, EnhancedFaceRecognition

# ════════════════════════════════════════════════════════════════════
#  SWITCH MODEL HERE
#  1 = Local face_recognition (dlib)  — Pi 3, no internet needed
#  2 = Claude Vision API              — requires internet
#  3 = Enhanced InsightFace/ArcFace   — Pi 5, not yet implemented
# ════════════════════════════════════════════════════════════════════
FR_MODEL = 2


# ── Config ────────────────────────────────────────────────────────────────────
_BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FACES_DIR    = os.path.join(_BASE_DIR, "faces")
CAMERA_INDEX  = 0
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "your-key-here")
CLAUDE_MODEL   = "claude-haiku-4-5-20251001"


class FacialRecognition:
    """
    Main facial recognition interface.
    Delegates to whichever model is selected via FR_MODEL.

    Switching models:
        Change FR_MODEL = 1 / 2 / 3 at the top of this file.
    """

    def __init__(self):
        self._model = self._load_model()
        print(f"FacialRecognition ready — Model {FR_MODEL}: {type(self._model).__name__}")

    def _load_model(self):
        if FR_MODEL == 1:
            return LocalFaceRecognition(
                faces_dir=_FACES_DIR,
                camera_index=CAMERA_INDEX
            )
        elif FR_MODEL == 2:
            return ClaudeFaceRecognition(
                faces_dir=_FACES_DIR,
                api_key=CLAUDE_API_KEY,
                camera_index=CAMERA_INDEX,
                model=CLAUDE_MODEL
            )
        elif FR_MODEL == 3:
            return EnhancedFaceRecognition(
                faces_dir=_FACES_DIR,
                camera_index=CAMERA_INDEX
            )
        else:
            raise ValueError(f"Invalid FR_MODEL: {FR_MODEL}. Must be 1, 2, or 3.")

    # ── Public API (same regardless of model) ─────────────────────────

    def enroll(self, name: str, image_path: str = None) -> bool:
        """
        Enroll a person.
        Pass image_path to enroll from file, or None to capture from camera.
        """
        print(f"Enrolling: {name}")
        result = self._model.enroll(name, image_path)
        print(f"Enrollment {'successful' if result else 'failed'}: {name}")
        return result

    def verify(self, name: str, override: bool = False, frame=None) -> bool:
        """Silent single-shot verification. Used in main dispenser flow."""
        print(f"Verifying: {name}")
        result = self._model.verify(name, override, frame=frame)
        print(f"Verification {'passed' if result else 'failed'}: {name}")
        return result

    def verify_live(self, name: str, override: bool = False) -> bool:
        """Live camera verification with visual display."""
        return self._model.verify_live(name, override)

    def list_enrolled(self) -> list:
        """Return all enrolled names."""
        return self._model.list_enrolled()

    def identify(self, frame=None) -> str | None:
        """
        Scan all enrolled residents and return the first match.
        Pass frame to reuse an already-captured image (avoids extra camera access).
        Returns the resident's name, or None if no match found.
        """
        for name in self.list_enrolled():
            if self.verify(name, frame=frame):
                return name
        return None

    def list_cameras(self):
        """Helper to find correct camera index."""
        import cv2
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                print(f"  [{i}] camera available")
            cap.release()