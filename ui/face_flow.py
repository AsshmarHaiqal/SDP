"""
ui/face_flow.py — Face scan and patient identification.

Runs entirely in a background thread.  Reports back via root.after() callbacks.

Flow:
    1. Sweep servo (face_tracking module or inline Haar cascade fallback).
    2. Lock on a face frame.
    3. POST frame to API /api/identify  (server mode)
       — or —  run local FacialRecognition (local mode)
    4. Call on_identified(patient_dict) or on_failed(reason).

The patient dict returned by on_identified uses the UNIFIED shape defined
in api_client.py, so both main.py and ui/ classes can consume it.
"""

import os
import time
import threading

import cv2
import numpy as np

# ── Servo sweep constants ─────────────────────────────────────────────────────
# Camera servo (ch 14) rests at 180° facing down at the tray.
# On "Ready to Collect", it sweeps UPWARD by reducing angle in 5° steps.
# It stops at 40° — the highest it can physically go.
# If no face is found by 40°, scan has failed.
_SCAN_HOME = 180   # resting position (down at tray)
_SCAN_MIN  = 40    # highest the camera can tilt (face-level)
_SCAN_STEP = 5     # degrees per step upward


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
    fr          : FacialRecognition instance or None  (local mode only)
    fr_mode     : "server" | "local"
    timeout_s   : seconds before giving up
    """

    def __init__(self, camera, api_client, servo, root, ft=None,
                 fr=None, fr_mode: str = "server", timeout_s: int = 15):
        self._camera     = camera
        self._api        = api_client
        self._servo      = servo
        self._root       = root
        self._ft         = ft
        self._fr         = fr
        self._fr_mode    = fr_mode
        self._timeout    = timeout_s
        self._stop_flag  = threading.Event()

    def stop(self):
        self._stop_flag.set()

    def start(self, on_identified, on_failed):
        """
        Kick off the face scan in a background thread.

        Callbacks are invoked on the tkinter main thread via root.after(0, ...).
          on_identified(patient: dict)   — unified patient dict
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
            self._root.after(
                0,
                lambda: on_failed(
                    "No face detected.\n"
                    "Please stand in front of the camera and try again."
                ),
            )
            return

        if self._stop_flag.is_set():
            return

        # Return camera to home position (180° — down at tray)
        if self._ft:
            self._ft.set_servo_angle(float(_SCAN_HOME))
        else:
            self._servo.set_servo_angle(14, float(_SCAN_HOME))

        # ── Identify ──────────────────────────────────────────────────────
        if self._fr_mode == "local":
            self._identify_local(frame, on_identified, on_failed)
        else:
            self._identify_server(frame, on_identified, on_failed)

    # ── Server-side identification ────────────────────────────────────────

    def _identify_server(self, frame, on_identified, on_failed):
        """POST frame to /api/identify via APIClient."""
        patient = self._api.identify_patient(frame)

        if patient and patient.get("ok") and patient.get("matched"):
            self._root.after(0, lambda p=patient: on_identified(p))
        else:
            reason = (
                "Face not recognised.\n"
                "Please try again or ask a carer for help."
            )
            self._root.after(0, lambda: on_failed(reason))

    # ── Local identification ──────────────────────────────────────────────

    def _identify_local(self, frame, on_identified, on_failed):
        """Run local FacialRecognition then fetch prescriptions."""
        if self._fr is None:
            self._root.after(
                0,
                lambda: on_failed(
                    "Local face recognition not initialised.\n"
                    "Please see a member of staff."
                ),
            )
            return

        matched_name = self._fr.identify(frame=frame)

        if not matched_name:
            self._root.after(
                0,
                lambda: on_failed(
                    "Face not recognised.\n"
                    "Please see a member of staff."
                ),
            )
            return

        try:
            patient_id = int(matched_name)
        except ValueError:
            self._root.after(
                0,
                lambda: on_failed(
                    "Identity error.\n"
                    "Please see a member of staff."
                ),
            )
            return

        # Fetch prescriptions — returns unified patient dict
        patient = self._api.get_patient_prescriptions(patient_id)
        if patient:
            self._root.after(0, lambda p=patient: on_identified(p))
        else:
            self._root.after(
                0,
                lambda: on_failed(
                    "No medication due right now.\n"
                    "Please see a member of staff."
                ),
            )

    # ── Face scanning ─────────────────────────────────────────────────────

    def _scan_for_face(self) -> np.ndarray | None:
        """
        Sweep camera servo from 180° (tray) upward to 40° (face level),
        stepping 5° at a time.  Returns the first frame containing a face,
        or None on timeout / no detection.

        When face_tracking module is loaded, uses _ft.cap and _ft.set_servo_angle
        directly (they own the hardware). Otherwise falls back to CameraManager
        and ServoController.
        """
        if self._ft:
            return self._scan_via_ft()
        return self._scan_inline()

    def _scan_via_ft(self) -> np.ndarray | None:
        """Use face_tracking's camera and servo directly."""
        # Safety check — face_tracking may have loaded (ServoKit OK) but
        # camera could be None if /dev/video0 was locked at import time.
        cap = getattr(self._ft, 'cap', None)
        if cap is None or not cap.isOpened():
            print("FaceFlow: face_tracking.cap not available — reopening camera")
            try:
                import cv2 as _cv2
                self._ft.cap = _cv2.VideoCapture('/dev/video0')
                time.sleep(0.5)
                cap = self._ft.cap
                if not cap.isOpened():
                    print("FaceFlow: camera reopen failed — falling back to inline scan")
                    return self._scan_inline()
            except Exception as e:
                print(f"FaceFlow: camera reopen error — {e}")
                return self._scan_inline()

        self._camera.pause_loop()

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        deadline = time.time() + self._timeout
        locked   = None
        angle    = float(_SCAN_HOME)

        # Start at home (180° — down at tray)
        self._ft.set_servo_angle(angle)
        time.sleep(0.3)

        # Sweep upward: 180 → 175 → 170 → … → 40
        while angle >= _SCAN_MIN and time.time() < deadline:
            if self._stop_flag.is_set():
                break

            angle -= _SCAN_STEP
            angle = max(float(_SCAN_MIN), angle)
            self._ft.set_servo_angle(angle)
            time.sleep(0.5)   # let servo physically settle

            if self._ft.cap is None or not self._ft.cap.isOpened():
                print("FaceFlow: camera lost during scan")
                break

            # Flush stale frames from the buffer — the camera may still
            # be showing the previous servo position
            for _ in range(3):
                self._ft.cap.read()

            ret, frame = self._ft.cap.read()
            if not ret or frame is None:
                continue

            # Update UI feed
            self._camera._latest = frame

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(faces) > 0:
                locked = frame.copy()
                print(f"FaceFlow: face detected at servo angle {angle:.0f}°")
                break

        # Return to home
        self._ft.set_servo_angle(float(_SCAN_HOME))

        self._camera.start_loop()
        return locked

    def _scan_inline(self) -> np.ndarray | None:
        """Fallback when face_tracking is not loaded — use CameraManager + ServoController."""
        self._camera.pause_loop()
        self._camera.open()

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        deadline = time.time() + self._timeout
        locked   = None
        angle    = float(_SCAN_HOME)

        self._servo.set_servo_angle(14, angle)
        time.sleep(0.3)

        while angle >= _SCAN_MIN and time.time() < deadline:
            if self._stop_flag.is_set():
                break

            angle -= _SCAN_STEP
            angle = max(float(_SCAN_MIN), angle)
            self._servo.set_servo_angle(14, angle)
            time.sleep(0.5)

            # Flush stale frames
            for _ in range(3):
                self._camera.read_frame()

            frame = self._camera.read_frame()
            if frame is None:
                continue

            self._camera._latest = frame

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(faces) > 0:
                locked = frame.copy()
                print(f"FaceFlow: face detected at servo angle {angle:.0f}°")
                break

        self._servo.set_servo_angle(14, float(_SCAN_HOME))

        self._camera.start_loop()
        return locked