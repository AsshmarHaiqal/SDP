"""
Base class all facial recognition models must implement.
"""
from abc import ABC, abstractmethod


class FaceRecognitionBase(ABC):

    @abstractmethod
    def enroll(self, name: str, image_path: str = None) -> bool:
        """
        Enroll a person by name.
        If image_path is None, capture from camera.
        Returns True on success.
        """

    @abstractmethod
    def verify(self, name: str, override: bool = False) -> bool:
        """
        Silent single-shot verification.
        Returns True if identity confirmed.
        """

    @abstractmethod
    def verify_live(self, name: str, override: bool = False) -> bool:
        """
        Live camera verification with visual display.
        Returns True if identity confirmed.
        """

    @abstractmethod
    def list_enrolled(self) -> list:
        """Return list of enrolled names."""