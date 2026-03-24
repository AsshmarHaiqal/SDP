"""
api_client.py — PillWheel ↔ SDP server integration
====================================================
Corrected against actual deployed API endpoints.

Real endpoint map:
  Admin login:           POST /api/verify/login
  Admin logout:          POST /api/verify/logout
  Patient prescriptions:  GET /api/patient/{id}/prescriptions
  All patients (admin):   GET /api/patient/getAllPatients
  Log intake:            POST /api/patient/{id}/intake
  Reduce stock:          POST /api/medicines/reduce
  Public ping:            GET /api/public/ping

Auth: cookie-based — POST /api/verify/login sets adminId,
      adminUsername, adminRoot session cookies. requests.Session
      stores and replays these automatically.
"""

import os
import re
import base64
import cv2
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

import requests

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

BASE_URL = os.getenv("SERVER_URL", "http://localhost:8080")

ADMIN_CREDENTIALS = {
    "username": os.getenv("PILLWHEEL_ADMIN_USER", "root"),
    "password": os.getenv("PILLWHEEL_ADMIN_PASS", "root"),
}

_TIMEOUT_FAST = 5
_TIMEOUT_SLOW = 20

# ── Medicine number → servo slot (1-based, channel = number - 1) ─────────────
_MEDICINE_NUMBER = {
    "VTM01":  1,
    "VTM02":  2,
    "VTM03":  3,
    "SUP01":  4,
    "MINMG":  5,
    "MINCA":  6,
    "MINZN":  7,
    "MINFE":  8,
    "SUP02":  9,
    "SUP03": 10,
    "SUP04": 11,
    "SUP05": 12,
    "MINK":  13,
    "SUP06": 14,
    "SUP07": 15,
    "SUP08": 16,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_pill_count(dosage: str) -> int:
    """
    Extract pill count from dosage string.
    '2 tablets' → 2,  '1000mg' → 1,  '' → 1
    """
    if not dosage:
        return 1
    m = re.match(r"^\s*(\d+)", dosage.strip())
    return int(m.group(1)) if m else 1


def _frame_to_jpeg_bytes(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _normalize_prescription(p: dict) -> dict:
    """
    Convert server prescription shape to what main.py expects.

    Server shape (GET /api/patient/{id}/prescriptions):
        { medicineId, medicineName, dosage, frequency, ... }

    main.py expects:
        { prescription_id, medicine_name, medicine_code,
          medicine_number, dosage, pill_count, scheduled_time }
    """
    medicine_id = p.get("medicineId", "")
    return {
        "prescription_id": p.get("id"),
        "medicine_name":   p.get("medicineName", "Unknown"),
        "medicine_code":   medicine_id,
        "medicine_number": _MEDICINE_NUMBER.get(medicine_id, 1),
        "dosage":          p.get("dosage", ""),
        "pill_count":      _parse_pill_count(p.get("dosage", "")),
        "scheduled_time":  p.get("scheduledTime", ""),
    }


# ── APIClient ─────────────────────────────────────────────────────────────────

class APIClient:
    """
    HTTP client for PillWheel ↔ SDP server.

    On init: pings server, then logs in via POST /api/verify/login.
    Session cookies are stored automatically by requests.Session.
    """

    def __init__(self):
        self._session = requests.Session()
        self._online  = False
        self._authed  = False

        print(f"APIClient: server = {BASE_URL}")
        self._check_online()
        if self._online:
            self._login()

    # ── Connectivity ──────────────────────────────────────────────────────────

    def _check_online(self) -> bool:
        try:
            r = self._session.get(
                f"{BASE_URL}/api/public/ping",
                timeout=_TIMEOUT_FAST,
            )
            self._online = r.ok
        except Exception:
            self._online = False
        print(f"APIClient: server {'online ✔' if self._online else 'OFFLINE'}")
        return self._online

    # ── Auth ──────────────────────────────────────────────────────────────────

    def _login(self) -> bool:
        """
        POST /api/verify/login
        Sets adminId, adminUsername, adminRoot session cookies on success.
        """
        try:
            r = self._session.post(
                f"{BASE_URL}/api/verify/login",
                json={
                    "username": ADMIN_CREDENTIALS["username"],
                    "password": ADMIN_CREDENTIALS["password"],
                },
                timeout=_TIMEOUT_FAST,
            )

            # Some Spring endpoints return empty body on success
            if not r.text.strip():
                self._authed = r.ok
                print(f"APIClient: login HTTP {r.status_code} "
                      f"{'✔' if r.ok else '✗'} (empty body)")
                return r.ok

            data = r.json()
            if r.ok and data.get("ok"):
                self._authed = True
                print(f"APIClient: logged in as '{data.get('username')}' "
                      f"(root={data.get('root', False)}) ✔")
                return True

            print(f"APIClient: login failed — {data}")
            return False

        except Exception as e:
            print(f"APIClient: login error — {e}")
            return False

    def _ensure_authed(self) -> bool:
        if self._authed:
            return True
        if not self._check_online():
            return False
        return self._login()

    def is_online(self) -> bool:
        return self._check_online()

    # ── Face image sync (FR_MODE=local) ──────────────────────────────────────

    def sync_faces_locally(self, faces_dir: str) -> int:
        """
        Download all patient face images and save as <patient_id>.jpg.
        Uses GET /api/patient/getAllPatients (requires admin cookie).
        Looks for faceData field (base64) on each patient record.

        Returns number of images saved.
        """
        if not self._ensure_authed():
            print("APIClient: sync_faces_locally — not authenticated")
            return 0

        os.makedirs(faces_dir, exist_ok=True)

        try:
            r = self._session.get(
                f"{BASE_URL}/api/patient/getAllPatients",
                timeout=_TIMEOUT_SLOW,
            )
            if not r.ok:
                print(f"APIClient: getAllPatients failed HTTP {r.status_code}")
                return 0

            data     = r.json()
            patients = data if isinstance(data, list) else data.get("patients", [])
            saved    = 0

            for p in patients:
                patient_id = p.get("id")
                username   = p.get("username") or str(patient_id)
                face_data  = p.get("faceData") or p.get("face_data")

                if not patient_id:
                    continue

                if not face_data:
                    print(f"  No face enrolled for {username} (id={patient_id})")
                    continue

                try:
                    # Strip data URI prefix if present
                    if "," in face_data:
                        face_data = face_data.split(",", 1)[1]

                    img_bytes = base64.b64decode(face_data)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if img is None:
                        print(f"  Could not decode image for {username}")
                        continue

                    # Save as patient_id.jpg — FR matches by ID
                    save_path = os.path.join(faces_dir, f"{patient_id}.jpg")
                    cv2.imwrite(save_path, img)
                    print(f"  Saved: {username} (id={patient_id}) → {save_path}")
                    saved += 1

                except Exception as e:
                    print(f"  Error saving face for {username}: {e}")

            print(f"APIClient: sync_faces_locally — {saved} image(s) saved")
            return saved

        except Exception as e:
            print(f"APIClient: sync_faces_locally error — {e}")
            return 0

    # ── Patient identification (server-side FR) ───────────────────────────────

    def identify_patient(self, frame: np.ndarray) -> dict | None:
        """
        No server-side machine FR endpoint exists.
        Use FR_MODE=local — Claude Vision identifies on-device,
        then call get_patient_prescriptions() with the matched patient ID.
        """
        print("APIClient: identify_patient — no server FR endpoint. "
              "Set FR_MODE=local.")
        return None

    # ── Patient prescriptions ─────────────────────────────────────────────────

    def get_patient_prescriptions(self, patient_id: int) -> dict | None:
        """
        GET /api/patient/{patientId}/prescriptions  (no auth required)

        Also fetches patient name via GET /api/admin/patients/{id} if authed.
        Returns patient dict shaped for main.py, or None.
        """
        if not self._check_online():
            print("APIClient: get_patient_prescriptions — server offline")
            return None

        # ── 1. Prescriptions (no auth needed) ─────────────────────────────────
        try:
            r = self._session.get(
                f"{BASE_URL}/api/patient/{patient_id}/prescriptions",
                timeout=_TIMEOUT_FAST,
            )
            if not r.ok:
                print(f"APIClient: prescriptions HTTP {r.status_code} "
                      f"for patient {patient_id}")
                return None

            data              = r.json()
            prescriptions_raw = data.get("prescriptions", [])

            if not prescriptions_raw:
                print(f"APIClient: no prescriptions for patient {patient_id}")
                return None

        except Exception as e:
            print(f"APIClient: get_patient_prescriptions error — {e}")
            return None

        # ── 2. Patient name (admin auth, best-effort) ──────────────────────────
        display_name = f"Patient {patient_id}"
        if self._ensure_authed():
            try:
                r2 = self._session.get(
                    f"{BASE_URL}/api/admin/patients/{patient_id}",
                    timeout=_TIMEOUT_FAST,
                )
                if r2.ok:
                    p     = r2.json()
                    first = p.get("firstName", "")
                    last  = p.get("lastName", "")
                    if first or last:
                        display_name = f"{first} {last}".strip()
            except Exception:
                pass

        prescriptions = [_normalize_prescription(p) for p in prescriptions_raw]

        patient = {
            "patient_id":    patient_id,
            "display_name":  display_name,
            "prescriptions": prescriptions,
        }

        print(f"APIClient: {display_name} — {len(prescriptions)} prescription(s)")
        return patient

    # ── Dispense logging ──────────────────────────────────────────────────────

    def log_dispense_result(
        self,
        patient_id: int,
        prescription_id: int,
        scheduled_time: str,
        status: str = "TAKEN",
        failure_reason: str | None = None,
    ) -> bool:
        """
        No /api/machine/dispense-result endpoint on server.

        Always writes to local audit log at data/audit/dispense_log.txt.
        On TAKEN status, also calls POST /api/patient/{id}/intake.
        """
        # ── Local audit log (always) ──────────────────────────────────────────
        ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = (
            f"{ts} | patient={patient_id} | rx={prescription_id} | "
            f"status={status}"
            + (f" | reason={failure_reason}" if failure_reason else "")
            + "\n"
        )
        try:
            audit_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data", "audit"
            )
            os.makedirs(audit_dir, exist_ok=True)
            with open(os.path.join(audit_dir, "dispense_log.txt"), "a") as f:
                f.write(log_line)
        except Exception as e:
            print(f"APIClient: local audit log error — {e}")

        print(f"APIClient: logged locally — {log_line.strip()}")

        # ── Server intake log (TAKEN only, best-effort) ───────────────────────
        if status.upper() != "TAKEN" or not self._check_online():
            return True

        taken_date = datetime.now().strftime("%Y-%m-%d")
        taken_time = datetime.now().strftime("%H:%M")
        if scheduled_time:
            try:
                dt = (datetime.fromisoformat(scheduled_time)
                      if "T" in scheduled_time
                      else datetime.strptime(scheduled_time, "%H:%M:%S"))
                taken_time = dt.strftime("%H:%M")
            except ValueError:
                pass

        try:
            r = self._session.post(
                f"{BASE_URL}/api/patient/{patient_id}/intake",
                json={
                    "medicineId": "",
                    "takenDate":  taken_date,
                    "takenTime":  taken_time,
                    "notes":      f"Dispensed by PillWheel (rx={prescription_id})",
                },
                timeout=_TIMEOUT_FAST,
            )
            if r.ok:
                print("APIClient: intake logged on server ✔")
            else:
                print(f"APIClient: intake log failed HTTP {r.status_code}")
        except Exception as e:
            print(f"APIClient: intake log error — {e}")

        return True

    # ── Stock reduction ───────────────────────────────────────────────────────

    def reduce_stock(self, medicine_name: str, quantity: int) -> bool:
        """
        POST /api/medicines/reduce — no auth required.
        """
        if not self._check_online():
            return False
        try:
            r = self._session.post(
                f"{BASE_URL}/api/medicines/reduce",
                json={"medicineName": medicine_name, "quantity": quantity},
                timeout=_TIMEOUT_FAST,
            )
            ok = r.ok and r.json().get("ok", False)
            if not ok:
                print(f"APIClient: reduce_stock failed — {r.text[:120]}")
            return ok
        except Exception as e:
            print(f"APIClient: reduce_stock error — {e}")
            return False

    # ── Patient images ────────────────────────────────────────────────────────

    def get_patient_images(self) -> list[dict]:
        """
        Returns raw patient list from GET /api/patient/getAllPatients.
        Used internally by sync_faces_locally.
        """
        if not self._ensure_authed():
            return []
        try:
            r = self._session.get(
                f"{BASE_URL}/api/patient/getAllPatients",
                timeout=_TIMEOUT_SLOW,
            )
            if not r.ok:
                return []
            data = r.json()
            return data if isinstance(data, list) else data.get("patients", [])
        except Exception as e:
            print(f"APIClient: get_patient_images error — {e}")
            return []