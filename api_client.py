"""
data/api_client.py
Handles all communication between the PillWheel machine and the SDP server.
"""

import requests
import base64
import cv2
from datetime import datetime

BASE_URL = "https://www.sdpgroup16.com/api"

# Machine logs in as root admin once at startup
ADMIN_CREDENTIALS = {
    "username": "root",      # update with real root credentials
    "password": "root"
}


class APIClient:

    def __init__(self):
        self._session = requests.Session()
        self._online  = False
        self._login()

    # ── Auth ──────────────────────────────────────────────────────────

    def _login(self):
        try:
            r = self._session.post(
                f"{BASE_URL}/verify/login",
                json=ADMIN_CREDENTIALS,
                timeout=5
            )
            if r.ok and r.json().get("ok"):
                self._online = True
                print("APIClient: logged in ✔")
            else:
                print(f"APIClient: login failed — {r.json()}")
        except Exception as e:
            self._online = False
            print(f"APIClient: server unreachable — {e}")

    def is_online(self) -> bool:
        try:
            r = self._session.get(f"{BASE_URL}/public/ping", timeout=3)
            self._online = r.ok
        except Exception:
            self._online = False
        return self._online

    # ── Facial recognition ────────────────────────────────────────────
    # Requires new endpoint on server:
    # POST /api/facial-recognition
    # Body: { "image": "<base64 jpg>" }
    # Response: {
    #   "ok": true,
    #   "patientId": 1,
    #   "displayName": "John Doe",
    #   "prescriptions": [
    #     {
    #       "prescriptionId": 1,
    #       "medicineId": "VTM01",
    #       "medicineName": "Vitamin C",
    #       "dosage": "1000mg",
    #       "frequency": "Once daily",
    #       "quantity": 2,
    #       "container": 3,        ← new column your team adds
    #       "instructions": "Take with food"
    #     }
    #   ],
    #   "nextCollection": "2026-03-18T09:00:00"
    # }

    def identify_patient(self, frame) -> dict | None:
        """
        Send captured frame to server for facial recognition.
        Returns patient + prescription dict, or None if unrecognised / offline.
        """
        if not self.is_online():
            return None
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64    = base64.b64encode(buf).decode("utf-8")

            r = self._session.post(
                f"{BASE_URL}/facial-recognition",
                json={"image": b64},
                timeout=15      # recognition may take a moment
            )
            data = r.json()
            if r.ok and data.get("ok"):
                return data
            print(f"APIClient: recognition failed — {data}")
            return None

        except Exception as e:
            print(f"APIClient: identify_patient error — {e}")
            return None

    # ── Post-dispense logging ─────────────────────────────────────────

    def log_intake(self, patient_id: int, medicine_id: str,
                   quantity: int, notes: str = "") -> bool:
        """Log a completed dispense event."""
        if not self.is_online():
            return False
        try:
            now = datetime.now()
            r = self._session.post(
                f"{BASE_URL}/patient/{patient_id}/intake",
                json={
                    "medicineId": medicine_id,
                    "takenDate":  now.strftime("%Y-%m-%d"),
                    "takenTime":  now.strftime("%H:%M"),
                    "notes":      notes or f"Dispensed × {quantity}"
                },
                timeout=5
            )
            return r.ok
        except Exception as e:
            print(f"APIClient: log_intake error — {e}")
            return False

    def reduce_stock(self, medicine_name: str, quantity: int) -> bool:
        """Reduce remaining pill count in database."""
        if not self.is_online():
            return False
        try:
            r = self._session.post(
                f"{BASE_URL}/medicines/reduce",
                json={"medicineName": medicine_name, "quantity": quantity},
                timeout=5
            )
            return r.ok
        except Exception as e:
            print(f"APIClient: reduce_stock error — {e}")
            return False

    def upload_audit_image(self, patient_id: int,
                           frame, timestamp: str) -> bool:
        """
        Upload tray image for audit log.
        Requires new endpoint on server:
        POST /api/patient/{patientId}/audit-image
        Body: multipart/form-data  field: "image" (JPEG)
              or JSON: { "image": "<base64>", "timestamp": "..." }
        """
        if not self.is_online():
            return False
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            b64    = base64.b64encode(buf).decode("utf-8")

            r = self._session.post(
                f"{BASE_URL}/patient/{patient_id}/audit-image",
                json={"image": b64, "timestamp": timestamp},
                timeout=10
            )
            return r.ok
        except Exception as e:
            print(f"APIClient: upload_audit_image error — {e}")
            return False