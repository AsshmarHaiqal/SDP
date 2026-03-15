"""
Model 3: Enhanced local facial recognition.
Placeholder — to be implemented with InsightFace/ArcFace on Pi 5.
"""
from .base import FaceRecognitionBase


class EnhancedFaceRecognition(FaceRecognitionBase):
    """
    Planned: InsightFace (ArcFace) implementation for Pi 5.
    Better accuracy for elderly users.
    Handles age variation, glasses, lighting changes.
    """

    def __init__(self, faces_dir: str, camera_index: int = 0):
        self.faces_dir    = faces_dir
        self.camera_index = camera_index
        print("NOTE: Enhanced model not yet implemented — use Model 1 or 2")

    def enroll(self, name: str, image_path: str = None) -> bool:
        raise NotImplementedError("Enhanced model coming in Pi 5 phase")

    def verify(self, name: str, override: bool = False) -> bool:
        raise NotImplementedError("Enhanced model coming in Pi 5 phase")

    def verify_live(self, name: str, override: bool = False) -> bool:
        raise NotImplementedError("Enhanced model coming in Pi 5 phase")

    def list_enrolled(self) -> list:
        raise NotImplementedError("Enhanced model coming in Pi 5 phase")