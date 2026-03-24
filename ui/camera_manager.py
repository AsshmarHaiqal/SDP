"""
ui/camera_manager.py — Centralised camera ownership and frame loop.

Handles the conflict between face_tracking (which opens VideoCapture at
module level) and the rest of the app.  When face_tracking is loaded, all
reads go through its cap.  Otherwise CameraManager owns its own cap.
"""

import threading
import time

import cv2
import numpy as np


class CameraManager:
    """
    Single owner of the camera feed.

    Usage:
        cam = CameraManager(face_tracking_module=_ft)
        cam.open()
        cam.start_loop()
        frame = cam.latest_frame   # always the most recent frame

        cam.stop_loop()
        cam.close()
    """

    def __init__(self, face_tracking_module=None):
        self._ft          = face_tracking_module
        self._cap         = None
        self._lock        = threading.Lock()
        self._loop_active = False
        self._latest      : np.ndarray | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the camera.  No-op when face_tracking owns the cap."""
        if self._ft:
            return
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                self._cap = cv2.VideoCapture("/dev/video0")
                time.sleep(0.5)

    def close(self) -> None:
        """Release the camera.  No-op when face_tracking owns the cap."""
        self._loop_active = False
        if self._ft:
            return
        time.sleep(0.12)
        with self._lock:
            if self._cap and self._cap.isOpened():
                self._cap.release()
            self._cap = None

    # ── Background loop ────────────────────────────────────────────────────────

    def start_loop(self) -> None:
        """Start a ~20-fps background thread filling self.latest_frame."""
        self._loop_active = True

        def _loop():
            while self._loop_active:
                frame = self.read_frame()
                if frame is not None:
                    self._latest = frame
                time.sleep(0.05)

        threading.Thread(target=_loop, daemon=True).start()

    def stop_loop(self) -> None:
        self._loop_active = False

    def pause_loop(self) -> None:
        """
        Temporarily pause the loop so another thread can have exclusive
        cap access (e.g. scan_for_face).  Call start_loop() to resume.
        """
        self._loop_active = False
        time.sleep(0.12)

    # ── Frame access ───────────────────────────────────────────────────────────

    @property
    def latest_frame(self) -> np.ndarray | None:
        return self._latest

    def read_frame(self) -> np.ndarray | None:
        """Thread-safe single frame read."""
        if self._ft:
            try:
                cap = self._ft.cap
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    return frame if ret else None
            except Exception:
                return None
        else:
            with self._lock:
                if self._cap and self._cap.isOpened():
                    ret, frame = self._cap.read()
                    return frame if ret else None
        return None