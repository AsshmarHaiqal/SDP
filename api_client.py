"""
api_client.py — PillWheel ↔ SDP server integration
====================================================
All machine-to-server calls go through this module.

Authentication: every request carries the  X-API-KEY  header.
The key is read from the environment variable MACHINE_API_KEY,
falling back to the default dev key used by the server.

Server base URL is read from  SERVER_URL  env-var so it can be
overridden without touching code:
    export SERVER_URL=http://192.168.1.50:8080
"""

import os
import re
import cv2
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

import requests

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

BASE_URL          = "https://www.sdpgroup16.com/api"
IDENTIFY_API_KEY  = os.getenv("PILLWHEEL_API_KEY", "")

ADMIN_CREDENTIALS = {
    "username": os.getenv("PILLWHEEL_ADMIN_USER", "root"),
    "password": os.getenv("PILLWHEEL_ADMIN_PASS", "root"),
}
_TIMEOUT_FAST = 5    # seconds — ping / stock / log
_TIMEOUT_SLOW = 20   # seconds — identify (Claude Vision round-trip)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_pill_count(dosage: str) -> int:
    """
    Best-effort extraction of an integer pill count from a dosage string.
    Examples:
        "2"          → 2
        "2 tablets"  → 2
        "500mg"      → 1  (no leading integer → default 1)
        ""           → 1
    """
    if not dosage:
        return 1
    m = re.match(r"^\s*(\d+)", dosage.strip())
    return int(m.group(1)) if m else 1


def _frame_to_jpeg_bytes(frame: np.ndarray) -> bytes:
    """Encode an OpenCV BGR frame to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _normalize_patient(data: dict) -> dict:
    """
    Convert the server MachineIdentifyResponse JSON into the patient dict
    format expected by main.py:

        {
            "patient_id":   int,
            "display_name": str,
            "prescriptions": [
                {
                    "prescription_id": int,
                    "medicine_name":   str,
                    "medicine_code":   str,   # e.g. "VTM01"
                    "medicine_number": int,   # 1-13 → servo channel = this - 1
                    "dosage":          str,
                    "pill_count":      int,   # parsed from dosage, default 1
                    "scheduled_time":  str,   # "HH:mm:ss"
                }
            ]
        }
    """
    prescriptions = []
    for p in data.get("prescriptions", []):
        prescriptions.append({
            "prescription_id": p.get("prescriptionId"),
            "medicine_name":   p.get("medicineName", "Unknown"),
            "medicine_code":   p.get("medicineCode", ""),
            "medicine_number": p.get("medicineNumber", 1),
            "dosage":          p.get("dosage", ""),
            "pill_count":      _parse_pill_count(p.get("dosage", "")),
            "scheduled_time":  p.get("scheduledTime", ""),
        })

    return {
        "patient_id":    data.get("patientId"),
        "display_name":  data.get("patientName", "Unknown"),
        "prescriptions": prescriptions,
    }


# ── APIClient ─────────────────────────────────────────────────────────────────

class APIClient:
    """
    Thin HTTP client for the PillWheel machine.

    All methods return a meaningful value on success and None / False on
    failure — callers never need to handle exceptions.
    """

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({"X-API-KEY": API_KEY})
        self._online = False
        print(f"APIClient: server = {BASE_URL}")
        self._check_online()

    # ── Connectivity ──────────────────────────────────────────────────────────

    def _check_online(self) -> bool:
        try:
            r = self._session.get(f"{BASE_URL}/api/public/ping", timeout=_TIMEOUT_FAST)
            self._online = r.ok
        except Exception:
            self._online = False
        status = "online ✔" if self._online else "OFFLINE"
        print(f"APIClient: server {status}")
        return self._online

    def is_online(self) -> bool:
        """Quick connectivity probe (call before any operation if desired)."""
        return self._check_online()

    # ── Facial recognition / patient identification ───────────────────────────

    def identify_patient(self, frame: np.ndarray) -> dict | None:
        """
        Send a captured BGR frame to the server for facial recognition.

        Returns a normalised patient dict on success, or None if:
          - the server is unreachable
          - no face matched
          - no prescriptions are currently due

        Server endpoint:  POST /api/machine/identify  (multipart, field "image")
        """
        if not self._check_online():
            print("APIClient: identify_patient — server offline")
            return None

        try:
            jpeg = _frame_to_jpeg_bytes(frame)
            r = self._session.post(
                f"{BASE_URL}/api/machine/identify",
                files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                timeout=_TIMEOUT_SLOW,
            )
            data = r.json()

            if not r.ok:
                print(f"APIClient: identify failed HTTP {r.status_code} — {data}")
                return None

            if not data.get("ok"):
                print(f"APIClient: identify — server error: {data.get('message')}")
                return None

            if not data.get("matched"):
                print("APIClient: identify — no matching patient")
                return None

            if not data.get("prescriptions"):
                print("APIClient: identify — patient found but no prescriptions due now")
                return None

            patient = _normalize_patient(data)
            print(f"APIClient: identified → {patient['display_name']} "
                  f"({len(patient['prescriptions'])} prescription(s) due)")
            return patient

        except Exception as e:
            print(f"APIClient: identify_patient error — {e}")
            return None

    # ── Post-dispense logging ─────────────────────────────────────────────────

    def log_dispense_result(
        self,
        patient_id: int,
        prescription_id: int,
        scheduled_time: str,
        status: str = "TAKEN",
        failure_reason: str | None = None,
    ) -> bool:
        """
        Log the outcome of a dispense attempt.

        status:         TAKEN | MISSED | FAILED | SKIPPED
        scheduled_time: ISO-8601 datetime, e.g. "2026-03-21T08:00:00"
                        If the Pi only has HH:mm:ss, today's date is prepended.

        Server endpoint:  POST /api/machine/dispense-result
        """
        if not self._check_online():
            print("APIClient: log_dispense_result — server offline, skipping log")
            return False

        # Normalise to full ISO-8601 if only a time string was given
        if scheduled_time and "T" not in scheduled_time:
            today = datetime.now().strftime("%Y-%m-%d")
            scheduled_time = f"{today}T{scheduled_time}"

        payload = {
            "patientId":      patient_id,
            "prescriptionId": prescription_id,
            "scheduledTime":  scheduled_time,
            "status":         status.upper(),
        }
        if failure_reason:
            payload["failureReason"] = failure_reason

        try:
            r = self._session.post(
                f"{BASE_URL}/api/machine/dispense-result",
                json=payload,
                timeout=_TIMEOUT_FAST,
            )
            ok = r.ok and r.json().get("ok", False)
            if not ok:
                print(f"APIClient: log_dispense_result failed — {r.json()}")
            return ok
        except Exception as e:
            print(f"APIClient: log_dispense_result error — {e}")
            return False

    def reduce_stock(self, medicine_name: str, quantity: int) -> bool:
        """
        Decrement stock for a medicine by name.
        Server endpoint:  POST /api/medicines/reduce
        """
        if not self._check_online():
            return False
        try:
            r = self._session.post(
                f"{BASE_URL}/api/medicines/reduce",
                json={"medicineName": medicine_name, "quantity": quantity},
                timeout=_TIMEOUT_FAST,
            )
            return r.ok
        except Exception as e:
            print(f"APIClient: reduce_stock error — {e}")
            return False

    # ── Startup: download enrolled face images ────────────────────────────────

    def get_patient_images(self) -> list[dict]:
        """
        Fetch all enrolled face images from the server.
        Useful for syncing a local FR cache if needed in future.
        Server endpoint:  GET /api/machine/patient-images
        """
        if not self._check_online():
            return []
        try:
            r = self._session.get(
                f"{BASE_URL}/api/machine/patient-images",
                timeout=_TIMEOUT_FAST,
            )
            data = r.json()
            if not r.ok or not data.get("ok"):
                print(f"APIClient: get_patient_images failed — {data.get('message')}")
                return []
            patients = data.get("patients", [])
            print(f"APIClient: {len(patients)} enrolled face(s) on server")
            return patients
        except Exception as e:
            print(f"APIClient: get_patient_images error — {e}")
            return []