"""
api_client.py — PillWheel ↔ SDP server integration
====================================================
Base URL: https://www.sdpgroup16.com

Endpoint map:
  Admin login:           POST /api/auth/admins/login
  Admin logout:          POST /api/auth/admins/logout
  Patient list:           GET /api/admin/patients
  Patient face images:    GET /api/admin/patients/images   (root)
  Patient search:         GET /api/admin/patients/search?q=
  Patient detail:         GET /api/admin/patients/{id}
  Patient prescriptions:  GET /api/patients/{patientId}/prescriptions
  Log intake:            POST /api/patients/{patientId}/intake
  Reduce stock:          POST /api/medicines/reduce
  Identify patient:      POST /api/machine/identify  (multipart, X-API-KEY header)
  Public ping:            GET /api/public/ping

Auth: cookie-based — POST /api/auth/admins/login sets adminId,
      adminUsername, adminRoot session cookies. requests.Session
      stores and replays these automatically.

UNIFIED PATIENT DICT SHAPE
---------------------------
Both identify_patient() and get_patient_prescriptions() return the same
shape so that main.py (asshmarhaiqal) and ui/ classes (winterscone) can
both consume the result without translation:

{
    # ── Identity fields (both repos) ──────────────────────────────────
    "ok":            True,
    "matched":       True,
    "patientId":     int,          # ui/ uses this
    "patient_id":    int,          # main.py uses this  (alias)
    "firstName":     str,
    "lastName":      str,
    "patientName":   str,          # "FirstName LastName"
    "display_name":  str,          # alias for patientName

    # ── Prescriptions ─────────────────────────────────────────────────
    "prescriptions": [
        {
            # winterscone / ui/ keys:
            "medicineName":    str,
            "medicineCode":    str,
            "quantity":        int,

            # asshmarhaiqal / main.py keys (aliases):
            "prescription_id": int | None,
            "medicine_name":   str,
            "medicine_code":   str,
            "medicine_number": int,
            "dosage":          str,
            "pill_count":      int,
            "scheduled_time":  str,
        }
    ],
}
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

BASE_URL = os.getenv("SERVER_URL", "https://www.sdpgroup16.com")

ADMIN_CREDENTIALS = {
    "username": os.getenv("PILLWHEEL_ADMIN_USER", "root"),
    "password": os.getenv("PILLWHEEL_ADMIN_PASS", "root"),
}

MACHINE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

_TIMEOUT_FAST = 5
_TIMEOUT_SLOW = 20

# ── Medicine ID → servo slot (1-based, channel = number - 1) ─────────────────
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


def _normalize_prescription(p: dict) -> dict:
    """
    Convert server prescription shape into a UNIFIED dict that carries
    BOTH sets of keys so main.py and ui/ classes work without translation.

    Server shape (GET /api/patients/{id}/prescriptions):
        { id, medicineId, medicineName, dosage, frequency, scheduledTime }

    Returned shape carries both naming conventions.
    """
    medicine_id   = p.get("medicineId", "")
    medicine_name = p.get("medicineName", "Unknown")
    dosage        = p.get("dosage", "")
    pill_count    = _parse_pill_count(dosage)
    med_number    = _MEDICINE_NUMBER.get(medicine_id, 1)
    scheduled     = p.get("scheduledTime", "")
    rx_id         = p.get("id")

    return {
        # ── winterscone / ui/ keys ────────────────────────────────────
        "medicineName":    medicine_name,
        "medicineCode":    medicine_id,
        "quantity":        pill_count,

        # ── asshmarhaiqal / main.py keys ──────────────────────────────
        "prescription_id": rx_id,
        "medicine_name":   medicine_name,
        "medicine_code":   medicine_id,
        "medicine_number": med_number,
        "dosage":          dosage,
        "pill_count":      pill_count,
        "scheduled_time":  scheduled,
    }


def _build_patient_dict(
    patient_id: int,
    first_name: str,
    last_name: str,
    prescriptions: list[dict],
    matched: bool = True,
) -> dict:
    """
    Build the unified patient dict consumed by both repos.
    """
    full_name = f"{first_name} {last_name}".strip() or f"Patient {patient_id}"
    return {
        "ok":            True,
        "matched":       matched,

        # winterscone keys
        "patientId":     patient_id,
        "firstName":     first_name,
        "lastName":      last_name,
        "patientName":   full_name,

        # asshmarhaiqal keys (aliases)
        "patient_id":    patient_id,
        "display_name":  full_name,

        "prescriptions": prescriptions,
    }


# ── APIClient ─────────────────────────────────────────────────────────────────

class APIClient:
    """
    HTTP client for PillWheel ↔ SDP server.

    On init: pings server, then logs in via POST /api/auth/admins/login.
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
        """POST /api/auth/admins/login — sets admin session cookies."""
        try:
            r = self._session.post(
                f"{BASE_URL}/api/auth/admins/login",
                json={
                    "username": ADMIN_CREDENTIALS["username"],
                    "password": ADMIN_CREDENTIALS["password"],
                },
                timeout=_TIMEOUT_FAST,
            )

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

    # ══════════════════════════════════════════════════════════════════════════
    #  identify_patient  —  POST /api/identify
    # ══════════════════════════════════════════════════════════════════════════

    def identify_patient(self, frame: np.ndarray) -> dict | None:
        """
        POST /api/machine/identify  (multipart/form-data)

        Server expects:
            Header:  X-API-KEY
            Part:    image  (JPEG file)

        Server returns MachineIdentifyResponse:
            { ok, matched, patientId, firstName, lastName, patientName,
              prescriptions: [...], message }

        Returns unified patient dict, or None on failure / no match.
        """
        if not self._check_online():
            print("APIClient: identify_patient — server offline")
            return None

        # Encode frame to JPEG bytes for multipart upload
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpg_bytes = buf.tobytes()
        except Exception as e:
            print(f"APIClient: identify_patient encode error — {e}")
            return None

        try:
            r = self._session.post(
                f"{BASE_URL}/api/machine/identify",
                headers={"X-API-KEY": MACHINE_API_KEY},
                files={"image": ("face.jpg", jpg_bytes, "image/jpeg")},
                timeout=_TIMEOUT_SLOW,
            )

            if not r.ok:
                print(f"APIClient: identify_patient HTTP {r.status_code}")
                return None

            data = r.json()

            if not data.get("ok") or not data.get("matched"):
                msg = data.get("message", "no match")
                print(f"APIClient: identify_patient — {msg}")
                return None

            # Server may return prescriptions directly, or we fetch them
            patient_id = data.get("patientId")
            first_name = data.get("firstName", "")
            last_name  = data.get("lastName", "")

            raw_rx = data.get("prescriptions", [])
            if raw_rx:
                prescriptions = [_normalize_prescription(p) for p in raw_rx]
            else:
                # Fallback: fetch prescriptions separately
                fallback = self.get_patient_prescriptions(patient_id)
                if fallback:
                    return fallback
                prescriptions = []

            if not prescriptions:
                print(f"APIClient: identify_patient — matched but no prescriptions")
                return None

            return _build_patient_dict(
                patient_id, first_name, last_name, prescriptions
            )

        except Exception as e:
            print(f"APIClient: identify_patient error — {e}")
            return None

    # ── Face image sync (FR_MODE=local) ──────────────────────────────────────

    def sync_faces_locally(self, faces_dir: str) -> int:
        """
        Download all patient face images and save as <patient_id>.jpg.

        Uses:
          GET /api/admin/patients/images  → [{username, image (data URI), contentType}]
          GET /api/admin/patients/search?q={username} → [{id, firstName, ...}]

        Saves each image as {patientId}.jpg so identify() can return
        an ID that get_patient_prescriptions() can look up directly.

        Returns number of images saved.
        """
        if not self._ensure_authed():
            print("APIClient: sync_faces_locally — not authenticated")
            return 0

        os.makedirs(faces_dir, exist_ok=True)

        try:
            r = self._session.get(
                f"{BASE_URL}/api/admin/patients/images",
                timeout=_TIMEOUT_SLOW,
            )
            if not r.ok:
                print(f"APIClient: /api/admin/patients/images failed HTTP {r.status_code}")
                return 0
            image_list = r.json()
        except Exception as e:
            print(f"APIClient: sync_faces_locally fetch error — {e}")
            return 0

        saved = 0
        for entry in image_list:
            username  = entry.get("username")
            image_uri = entry.get("image")

            if not username:
                continue
            if not image_uri:
                print(f"  No face image for username={username}")
                continue

            patient_id = self._resolve_patient_id(username)
            if patient_id is None:
                print(f"  Could not resolve patient ID for username={username}")
                continue

            try:
                b64_data  = image_uri.split(",", 1)[1] if "," in image_uri else image_uri
                img_bytes = base64.b64decode(b64_data)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None:
                    print(f"  Could not decode image for {username}")
                    continue

                save_path = os.path.join(faces_dir, f"{patient_id}.jpg")
                cv2.imwrite(save_path, img)
                print(f"  Saved: {username} (id={patient_id}) → {save_path}")
                saved += 1

            except Exception as e:
                print(f"  Error saving face for {username}: {e}")

        print(f"APIClient: sync_faces_locally — {saved} image(s) saved")
        return saved

    def _resolve_patient_id(self, username: str) -> int | None:
        """
        GET /api/admin/patients/search?q={username}
        Returns the first matching patient's numeric ID, or None.
        """
        try:
            r = self._session.get(
                f"{BASE_URL}/api/admin/patients/search",
                params={"q": username},
                timeout=_TIMEOUT_FAST,
            )
            if not r.ok:
                return None
            results = r.json()
            if results:
                return results[0].get("id")
        except Exception as e:
            print(f"APIClient: _resolve_patient_id error — {e}")
        return None

    # ── Patient prescriptions ─────────────────────────────────────────────────

    def get_patient_prescriptions(self, patient_id: int) -> dict | None:
        """
        GET /api/patients/{patientId}/prescriptions  (no auth required)

        Also fetches patient display name via GET /api/admin/patients/{id}
        if authenticated (best-effort).

        Returns unified patient dict, or None.
        """
        if not self._check_online():
            print("APIClient: get_patient_prescriptions — server offline")
            return None

        # 1. Prescriptions
        try:
            r = self._session.get(
                f"{BASE_URL}/api/patients/{patient_id}/prescriptions",
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

        # 2. Patient display name (admin auth, best-effort)
        first_name = ""
        last_name  = ""
        if self._ensure_authed():
            try:
                r2 = self._session.get(
                    f"{BASE_URL}/api/admin/patients/{patient_id}",
                    timeout=_TIMEOUT_FAST,
                )
                if r2.ok:
                    p          = r2.json()
                    first_name = p.get("firstName", "")
                    last_name  = p.get("lastName", "")
            except Exception:
                pass

        prescriptions = [_normalize_prescription(p) for p in prescriptions_raw]

        patient = _build_patient_dict(
            patient_id, first_name, last_name, prescriptions
        )

        print(f"APIClient: {patient['display_name']} — {len(prescriptions)} prescription(s)")
        return patient

    # ── Dispense logging ──────────────────────────────────────────────────────

    def log_dispense_result(
        self,
        patient_id: int,
        prescription_id: int,
        medicine_id: str = "",
        scheduled_time: str = "",
        status: str = "TAKEN",
        failure_reason: str | None = None,
    ) -> bool:
        """
        Always writes to local audit log at data/audit/dispense_log.txt.
        On TAKEN status, also calls POST /api/patients/{id}/intake.
        """
        ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = (
            f"{ts} | patient={patient_id} | rx={prescription_id} | "
            f"medicine={medicine_id} | status={status}"
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
                f"{BASE_URL}/api/patients/{patient_id}/intake",
                json={
                    "medicineId": medicine_id,
                    "takenDate":  taken_date,
                    "takenTime":  taken_time,
                    "notes":      f"Dispensed by PillWheel (rx={prescription_id})",
                },
                timeout=_TIMEOUT_FAST,
            )
            if r.ok:
                print("APIClient: intake logged on server ✔")
            else:
                print(f"APIClient: intake log failed HTTP {r.status_code} — {r.text[:120]}")
        except Exception as e:
            print(f"APIClient: intake log error — {e}")

        return True

    # ── Stock reduction ───────────────────────────────────────────────────────

    def reduce_stock(self, medicine_name: str, quantity: int) -> bool:
        """POST /api/medicines/reduce — no auth required."""
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

    # ── Patient list ──────────────────────────────────────────────────────────

    def get_all_patients(self) -> list[dict]:
        """
        GET /api/admin/patients — returns List<PatientSummaryDto>.
        Requires admin auth.
        """
        if not self._ensure_authed():
            return []
        try:
            r = self._session.get(
                f"{BASE_URL}/api/admin/patients",
                timeout=_TIMEOUT_SLOW,
            )
            if not r.ok:
                print(f"APIClient: get_all_patients HTTP {r.status_code}")
                return []
            return r.json()
        except Exception as e:
            print(f"APIClient: get_all_patients error — {e}")
            return []