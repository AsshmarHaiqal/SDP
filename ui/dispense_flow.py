"""
ui/dispense_flow.py — Pill dispensing logic.

Runs entirely in a background thread.
Callbacks post results back to the tkinter thread via root.after(0, ...).

Reads the UNIFIED patient dict from api_client.py.  Each prescription
carries BOTH naming conventions, so this class uses the winterscone keys
as its primary access pattern but everything works regardless.

Medicine code → dispenser slot mapping:
    slot = int(medicine_code) - 1
    e.g. medicine_code "01" → servo channel 0 (PCA9685 ch 0)
         medicine_code "VTM01" → looked up via medicine_number - 1
"""

import os
import time
import threading
from datetime import datetime

import cv2

MAX_PILL_RETRY = 3


class DispenseFlow:
    """
    Encapsulates one complete dispense cycle for a single patient.

    Parameters
    ----------
    servo       : ServoController
    pill_rec    : PillRecogniser
    tray_sweep  : callable (no args)
    sound       : SoundActuator
    camera      : CameraManager
    api_client  : APIClient
    root        : tk.Tk  (for thread-safe UI callbacks)
    audit_dir   : str    (path to write audit images)
    """

    def __init__(self, servo, pill_rec, tray_sweep, sound, camera,
                 api_client, root, audit_dir: str):
        self._servo      = servo
        self._pill_rec   = pill_rec
        self._tray_sweep = tray_sweep
        self._sound      = sound
        self._camera     = camera
        self._api        = api_client
        self._root       = root
        self._audit_dir  = audit_dir
        self._stop_flag  = threading.Event()
        os.makedirs(audit_dir, exist_ok=True)

    def stop(self):
        self._stop_flag.set()

    def start(self, patient: dict, on_status, on_complete, on_error):
        """
        Begin dispensing in a background thread.

          on_status(msg: str)                       — live status label updates
          on_complete(patient, rx, timestamp: str)   — success
          on_error(msg: str)                        — failure
        """
        self._stop_flag.clear()
        threading.Thread(
            target=self._run,
            args=(patient, on_status, on_complete, on_error),
            daemon=True,
        ).start()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self, patient, on_status, on_complete, on_error):
        rx = patient["prescriptions"][0]

        # Read from unified dict — try winterscone keys first, fall back to
        # asshmarhaiqal keys so it works no matter which repo produced the dict.
        expected = rx.get("quantity") or rx.get("pill_count", 1)
        med_name = rx.get("medicineName") or rx.get("medicine_name", "Unknown")
        med_code = rx.get("medicineCode") or rx.get("medicine_code", "1")
        pid      = patient.get("patientId") or patient.get("patient_id")

        # Determine servo slot
        # If medicine_code is numeric (e.g. "01"), slot = int(code) - 1
        # If it's an alpha code (e.g. "VTM01"), use medicine_number - 1
        try:
            slot = max(0, int(med_code) - 1)
        except ValueError:
            med_number = rx.get("medicine_number", 1)
            slot = max(0, int(med_number) - 1)

        def status(msg):
            self._root.after(0, lambda m=msg: on_status(m))

        # ── 1. Wait until tray is clear ───────────────────────────────────────
        while not self._stop_flag.is_set():
            frame = self._camera.latest_frame
            if frame is not None:
                count, _ = self._pill_rec.count_pills(frame=frame)
                if count == 0:
                    break
            status("Please clear the tray before collecting")
            self._sound.speak("Please clear the tray before collecting")
            time.sleep(3)

        if self._stop_flag.is_set():
            return

        status("Tray clear. Starting dispense.")
        time.sleep(1)

        # ── 2. Dispense pills one at a time ───────────────────────────────────
        audit_frame = None

        for i in range(expected):
            if self._stop_flag.is_set():
                return

            status(f"Dispensing pill {i+1} of {expected}…")
            self._servo.rotate_dispenser(slot)
            time.sleep(1.5)

            verified = False
            for attempt in range(MAX_PILL_RETRY):
                frame = self._camera.latest_frame
                if frame is None:
                    time.sleep(0.3)
                    continue
                audit_frame        = frame.copy()
                count, description = self._pill_rec.count_pills(frame=frame)

                if count == i + 1:
                    status(f"Pill {i+1} of {expected} detected ✓  ({description})")
                    verified = True
                    break

                status(
                    f"Expected {i+1}, detected {count} "
                    f"— retrying ({attempt+1}/{MAX_PILL_RETRY})…"
                )
                time.sleep(1.0)

            if not verified:
                self._sound.error()
                idx = i + 1
                self._root.after(
                    0,
                    lambda i=idx: on_error(
                        f"Pill {i} not confirmed after {MAX_PILL_RETRY} attempts.\n"
                        "Please call for assistance."
                    ),
                )
                return

        # ── 3. Save audit image ───────────────────────────────────────────────
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        audit_path = None
        if audit_frame is not None:
            audit_path = os.path.join(self._audit_dir, f"{pid}_{timestamp}.jpg")
            cv2.imwrite(audit_path, audit_frame)

        # ── 4. Sweep tray ─────────────────────────────────────────────────────
        status("Dispensing complete. Collecting medication…")
        self._tray_sweep()

        first_name = (
            patient.get("firstName")
            or patient.get("display_name", "there").split()[0]
        )
        self._sound.collected()
        self._sound.speak(f"Have a lovely day, {first_name}")

        # ── 5. Log to server ──────────────────────────────────────────────────
        self._api.log_dispense_result(
            patient_id      = pid,
            prescription_id = rx.get("prescription_id") or rx.get("id", 0),
            medicine_id     = med_code,
            scheduled_time  = rx.get("scheduled_time") or rx.get("scheduledTime", ""),
            status          = "TAKEN",
        )
        self._api.reduce_stock(med_name, expected)

        # ── 6. Callback to UI ────────────────────────────────────────────────
        self._root.after(
            0,
            lambda: on_complete(patient, rx, timestamp),
        )