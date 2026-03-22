"""
app/face_flow.py — Face scan and patient identification.

Runs entirely in a background thread.  Reports back via root.after() callbacks.

Flow:
    1. Sweep servo (face_tracking module or inline Haar cascade fallback).
    2. Lock on a face frame.
    3. POST frame to API /api/identify.
    4. Call on_identified(patient_dict) or on_failed(reason).
"""

import time
import threading

import cv2
import numpy as np

# Inline sweep constants (used when face_tracking is not available)
_SCAN_HOME = 180
_SCAN_MIN  = 0
_SCAN_STEP = 10


class FaceFlow:
    """
    Encapsulates the face-scan + identify step of the dispense flow.

    Parameters
    ----------
    camera      : CameraManager
    api_client  : APIClient
    servo       : ServoController  (used for inline fallback camera pan)
    root        : tk.Tk            (for root.after() callbacks to UI thread)
    ft          : face_tracking module or None
    timeout_s   : seconds before giving up
    """

    def __init__(self, camera, api_client, servo, root, ft=None, timeout_s: int = 15):
        self._camera     = camera
        self._api        = api_client
        self._servo      = servo
        self._root       = root
        self._ft         = ft
        self._timeout    = timeout_s
        self._stop_flag  = threading.Event()

    def stop(self):
        self._stop_flag.set()

    def start(self, on_identified, on_failed):
        """
        Kick off the face scan in a background thread.

        Callbacks are invoked on the tkinter main thread via root.after(0, ...).
          on_identified(patient: dict)
          on_failed(reason: str)
        """
        self._stop_flag.clear()
        threading.Thread(
            target=self._run,
            args=(on_identified, on_failed),
            daemon=True,
        ).start()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self, on_identified, on_failed):
        frame = self._scan_for_face()

        if frame is None:
            self._root.after(0, lambda: on_failed("No face detected.\nPlease stand in front of the camera and try again."))
            return

        if self._stop_flag.is_set():
            return

        # Return camera to default position
        if self._ft:
            try:
                self._ft.set_servo_angle(self._ft.DEFAULT_ANGLE_CAMERA)
            except Exception:
                pass

        patient = self._api.identify_patient(frame)

        if patient and patient.get("ok") and patient.get("matched"):
            self._root.after(0, lambda p=patient: on_identified(p))
        else:
            reason = "Face not recognised.\nPlease try again or ask a carer for help."
            self._root.after(0, lambda: on_failed(reason))

    # ── Scanning strategies ───────────────────────────────────────────────────

    def _scan_for_face(self) -> np.ndarray | None:
        if self._ft:
            return self._scan_via_ft()
        return self._scan_inline()

    def _scan_via_ft(self) -> np.ndarray | None:
        """Use face_tracking's servo + Haar cascade."""
        # Pause camera loop so scan_for_face has exclusive cap access
        self._camera.pause_loop()

        deadline = time.time() + self._timeout
        locked   = None

        while time.time() < deadline:
            if self._stop_flag.is_set():
                break

            face, frame, _angle = self._ft.scan_for_face(self._ft.DEFAULT_ANGLE_CAMERA)

            if face is not None and frame is not None:
                self._camera._latest = frame.copy()
                locked = frame.copy()
                break

            time.sleep(0.3)

        # Restart camera loop for the dispensing live feed
        self._camera.start_loop()
        return locked

    def _scan_inline(self) -> np.ndarray | None:
        """Fallback: OpenCV Haar cascade + ServoController angle sweep."""
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        angle     = float(_SCAN_HOME)
        direction = -1
        deadline  = time.time() + self._timeout

        while time.time() < deadline:
            if self._stop_flag.is_set():
                return None

            frame = self._camera.latest_frame
            if frame is not None:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                if len(faces) > 0:
                    return frame.copy()

            # Step servo
            if direction == -1:
                angle = max(float(_SCAN_MIN), angle - _SCAN_STEP)
                self._servo.set_servo_angle(14, angle)
                if angle <= _SCAN_MIN:
                    direction = 1
            else:
                angle = min(float(_SCAN_HOME), angle + _SCAN_STEP)
                self._servo.set_servo_angle(14, angle)
                if angle >= _SCAN_HOME:
                    direction = -1

        return None