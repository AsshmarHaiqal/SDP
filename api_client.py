"""
api_client.py — PillWheel ↔ SDP server integration
====================================================
Fixed version — aligned with actual server API (cookie-based auth).

Key changes from original:
  - Auth is cookie-based (POST /api/auth/admins/login), not X-API-KEY header
  - SERVER_URL read from env var (defaults to localhost:8080)
  - Added sync_faces_locally()       — downloads face images for FR_MODE=local
  - Added get_patient_prescriptions() — GET /api/patients/{id}/prescriptions
  - Fixed get_patient_images()        — correct path /api/admin/patients/images
  - log_dispense_result()             — uses intake endpoint as substitute
                                        (no /api/machine/dispense-result exists)
  - reduce_stock()                    — unchanged, endpoint exists and needs no auth

Usage:
    export SERVER_URL=http://localhost:8080          # or your Pi's server IP
    export PILLWHEEL_ADMIN_USER=root
    export PILLWHEEL_ADMIN_PASS=root
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

_TIMEOUT_FAST = 5    # seconds — ping / stock / log
_TIMEOUT_SLOW = 20   # seconds — image downloads / identify

# ── Medicine number map (medicineId → servo slot number) ─────────────────────
# Derived from MedicineType enum in server docs.
# medicine_number is 1-based; servo channel = medicine_number - 1
_MEDICINE_NUMBER = {
    "VTM01":  1,   # Vitamin C
    "VTM02":  2,   # Vitamin E
    "VTM03":  3,   # Vitamin B6
    "SUP01":  4,   # Omega-3 Fish Oil
    "MINMG":  5,   # Magnesium
    "MINCA":  6,   # Calcium
    "MINZN":  7,   # Zinc
    "MINFE":  8,   # Iron
    "SUP02":  9,   # Probiotics
    "SUP03": 10,   # Turmeric
    "SUP04": 11,   # CoQ10
    "SUP05": 12,   # Ashwagandha
    "MINK":  13,   # Potassium
    "SUP06": 14,   # Ginkgo Biloba
    "SUP07": 15,   # Milk Thistle
    "SUP08": 16,   # L-Theanine
}


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


def _normalize_prescription(p: dict) -> dict:
    """
    Convert a server prescription dict into the format expected by main.py.

    Server shape (from GET /api/patients/{id}/prescriptions):
        {
            "medicineId":   "VTM01",
            "medicineName": "Vitamin C",
            "dosage":       "1000mg",
            "frequency":    "Once daily"
        }

    Target shape (what main.py reads):
        {
            "prescription_id": int | None,
            "medicine_name":   str,
            "medicine_code":   str,
            "medicine_number": int,   # 1-13, servo channel = this - 1
            "dosage":          str,
            "pill_count":      int,
            "scheduled_time":  str,
        }
    """
    medicine_id = p.get("medicineId", "")
    return {
        "prescription_id": p.get("id"),                          # may be absent
        "medicine_name":   p.get("medicineName", "Unknown"),
        "medicine_code":   medicine_id,
        "medicine_number": _MEDICINE_NUMBER.get(medicine_id, 1), # default slot 1
        "dosage":          p.get("dosage", ""),
        "pill_count":      _parse_pill_count(p.get("dosage", "")),
        "scheduled_time":  p.get("scheduledTime", ""),           # not in current API
    }


# ── APIClient ─────────────────────────────────────────────────────────────────

class APIClient:
    """
    Thin HTTP client for the PillWheel machine.

    Auth flow:
        On __init__, calls POST /api/auth/admins/login.
        The session stores the returned cookies automatically.
        All subsequent /api/admin/** requests carry those cookies.

    All methods return a meaningful value on success and None / False on
    failure — callers never need to handle exceptions.
    """

    def __init__(self):
        self._session  = requests.Session()
        self._online   = False
        self._authed   = False

        print(f"APIClient: server = {BASE_URL}")
        self._check_online()
        if self._online:
            self._login()

    # ── Connectivity & auth ───────────────────────────────────────────────────

    def _check_online(self) -> bool:
        """Ping the public health-check endpoint (no auth needed)."""
        try:
            r = self._session.get(
                f"{BASE_URL}/api/public/ping",
                timeout=_TIMEOUT_FAST,
            )
            self._online = r.ok
        except Exception:
            self._online = False

        status = "online ✔" if self._online else "OFFLINE"
        print(f"APIClient: server {status}")
        return self._online

    def _login(self) -> bool:
        """
        Authenticate with the server using admin credentials.
        On success, session cookies are stored automatically by requests.
        Returns True on success.
        """
        try:
            r = self._session.post(
                f"{BASE_URL}/api/auth/admins/login",
                json={
                    "username": ADMIN_CREDENTIALS["username"],
                    "password": ADMIN_CREDENTIALS["password"],
                },
                timeout=_TIMEOUT_FAST,
            )
            data = r.json()
            if r.ok and data.get("ok"):
                self._authed = True
                root = data.get("root", False)
                print(f"APIClient: logged in as '{data.get('username')}' "
                      f"(root={root}) ✔")
                return True
            print(f"APIClient: login failed — {data}")
            return False
        except Exception as e:
            print(f"APIClient: login error — {e}")
            return False

    def _ensure_authed(self) -> bool:
        """Re-login if session has expired. Returns True if ready."""
        if self._authed:
            return True
        if not self._check_online():
            return False
        return self._login()

    def is_online(self) -> bool:
        return self._check_online()

    # ── Face image sync (for FR_MODE=local) ──────────────────────────────────

    def sync_faces_locally(self, faces_dir: str) -> int:
        """
        Download all enrolled patient face images from the server and save
        them to faces_dir as <username>.jpg files.

        Uses GET /api/admin/patients/images (requires root admin cookie).
        Returns the number of images successfully saved.

        Called by main.py._sync_and_enroll() when FR_MODE=local.
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
                print(f"APIClient: sync_faces_locally failed HTTP {r.status_code}")
                return 0

            patients = r.json()  # List[{username, image, contentType}]
            saved = 0

            for p in patients:
                username = p.get("username")
                image_b64 = p.get("image")   # "data:image/png;base64,..." or None

                if not username or not image_b64:
                    continue

                # Strip the data URI prefix if present
                if "," in image_b64:
                    image_b64 = image_b64.split(",", 1)[1]

                try:
                    img_bytes = base64.b64decode(image_b64)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if img is None:
                        print(f"  Could not decode image for {username}")
                        continue

                    save_path = os.path.join(faces_dir, f"{username}.jpg")
                    cv2.imwrite(save_path, img)
                    print(f"  Saved face: {username} → {save_path}")
                    saved += 1

                except Exception as e:
                    print(f"  Error saving face for {username}: {e}")

            print(f"APIClient: sync_faces_locally — {saved}/{len(patients)} images saved")
            return saved

        except Exception as e:
            print(f"APIClient: sync_faces_locally error — {e}")
            return 0

    # ── Patient identification (server-side FR) ───────────────────────────────

    def identify_patient(self, frame: np.ndarray) -> dict | None:
        """
        NOTE: The server does not have a /api/machine/identify endpoint.
        This method is a no-op stub — use FR_MODE=local instead.

        When FR_MODE=local, main.py calls fr.identify() directly and then
        calls get_patient_prescriptions() to fetch the patient data.
        """
        print("APIClient: identify_patient — server-side FR not available. "
              "Use FR_MODE=local.")
        return None

    # ── Patient prescriptions ─────────────────────────────────────────────────

    def get_patient_prescriptions(self, patient_id: int) -> dict | None:
        """
        Fetch prescriptions for a known patient ID.
        Uses GET /api/patients/{id}/prescriptions (no auth required).

        Also fetches patient name via GET /api/admin/patients/{id} (admin auth).

        Returns a patient dict in the shape expected by main.py:
            {
                "patient_id":    int,
                "display_name":  str,
                "prescriptions": [ { prescription fields... } ]
            }
        Returns None if the patient has no prescriptions or is not found.
        """
        if not self._check_online():
            print("APIClient: get_patient_prescriptions — server offline")
            return None

        # ── 1. Fetch prescriptions (no auth needed) ───────────────────────────
        try:
            r = self._session.get(
                f"{BASE_URL}/api/patients/{patient_id}/prescriptions",
                timeout=_TIMEOUT_FAST,
            )
            if not r.ok:
                print(f"APIClient: prescriptions failed HTTP {r.status_code} "
                      f"for patient {patient_id}")
                return None

            data = r.json()
            prescriptions_raw = data.get("prescriptions", [])

            if not prescriptions_raw:
                print(f"APIClient: no prescriptions for patient {patient_id}")
                return None

        except Exception as e:
            print(f"APIClient: get_patient_prescriptions error — {e}")
            return None

        # ── 2. Fetch patient name (admin auth, best-effort) ───────────────────
        display_name = f"Patient {patient_id}"
        if self._ensure_authed():
            try:
                r2 = self._session.get(
                    f"{BASE_URL}/api/admin/patients/{patient_id}",
                    timeout=_TIMEOUT_FAST,
                )
                if r2.ok:
                    p = r2.json()
                    first = p.get("firstName", "")
                    last  = p.get("lastName", "")
                    if first or last:
                        display_name = f"{first} {last}".strip()
            except Exception:
                pass   # name fetch failure is non-fatal

        prescriptions = [_normalize_prescription(p) for p in prescriptions_raw]

        patient = {
            "patient_id":    patient_id,
            "display_name":  display_name,
            "prescriptions": prescriptions,
        }

        print(f"APIClient: patient {display_name} — "
              f"{len(prescriptions)} prescription(s)")
        return patient

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

        NOTE: The server has no /api/machine/dispense-result endpoint.
        This method:
          - Always writes to a local audit log file (data/audit/dispense_log.txt)
          - On TAKEN status, also calls POST /api/patients/{id}/intake
            as the closest available server-side equivalent

        status: TAKEN | MISSED | FAILED | SKIPPED
        """
        # ── Local audit log (always written regardless of server) ─────────────
        timestamp = datetime.now().strftime("%Y-%m-%d %Human:%M:%S")
        log_line  = (
            f"{timestamp} | patient={patient_id} | rx={prescription_id} | "
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

        print(f"APIClient: dispense logged locally — {log_line.strip()}")

        # ── Server-side intake log (TAKEN only, best-effort) ──────────────────
        if status.upper() != "TAKEN":
            return True   # non-TAKEN events only go to local log for now

        if not self._check_online():
            return True   # local log written, server just unavailable

        # Normalise scheduled_time to date/time parts for intake endpoint
        taken_date = datetime.now().strftime("%Y-%m-%d")
        taken_time = datetime.now().strftime("%H:%M")
        if scheduled_time:
            try:
                if "T" in scheduled_time:
                    dt = datetime.fromisoformat(scheduled_time)
                else:
                    dt = datetime.strptime(scheduled_time, "%H:%M:%S")
                taken_time = dt.strftime("%H:%M")
            except ValueError:
                pass

        try:
            r = self._session.post(
                f"{BASE_URL}/api/patients/{patient_id}/intake",
                json={
                    "medicineId": "",     # not available at this point
                    "takenDate":  taken_date,
                    "takenTime":  taken_time,
                    "notes":      f"Dispensed by PillWheel (rx={prescription_id})",
                },
                timeout=_TIMEOUT_FAST,
            )
            if r.ok:
                print("APIClient: intake logged on server ✔")
            else:
                print(f"APIClient: intake log failed HTTP {r.status_code} "
                      f"— {r.text[:120]}")
        except Exception as e:
            print(f"APIClient: intake log error — {e}")

        return True   # local log always succeeds

    # ── Medicine stock ────────────────────────────────────────────────────────

    def reduce_stock(self, medicine_name: str, quantity: int) -> bool:
        """
        Decrement stock for a medicine by name.
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
                print(f"APIClient: reduce_stock failed — {r.json()}")
            return ok
        except Exception as e:
            print(f"APIClient: reduce_stock error — {e}")
            return False

    # ── Patient images (admin endpoint) ──────────────────────────────────────

    def get_patient_images(self) -> list[dict]:
        """
        Fetch all enrolled patient face images from the server.
        GET /api/admin/patients/images — requires root admin cookie.

        Returns list of:
            { "username": str, "image": "data:image/...;base64,...", "contentType": str }
        """
        if not self._ensure_authed():
            return []
        try:
            r = self._session.get(
                f"{BASE_URL}/api/admin/patients/images",
                timeout=_TIMEOUT_SLOW,
            )
            if not r.ok:
                print(f"APIClient: get_patient_images failed HTTP {r.status_code}")
                return []
            patients = r.json()
            print(f"APIClient: {len(patients)} enrolled face(s) on server")
            return patients
        except Exception as e:
            print(f"APIClient: get_patient_images error — {e}")
            return []