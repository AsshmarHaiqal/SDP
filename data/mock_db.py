"""
data/mock_db.py — Mock patient database for PillWheel.

Stores two sample residents with prescriptions. Pure Python, no external
database dependencies. Generates placeholder face images on first run using
OpenCV so the system can be demoed without real photos.
"""

import os
import json
from datetime import datetime

try:
    import cv2
    import numpy as np
    _CV2 = True
except ImportError:
    _CV2 = False

_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_FACES_DIR = os.path.join(_BASE_DIR, "faces")
_AUDIT_DIR = os.path.join(_BASE_DIR, "audit")
_LOG_FILE  = os.path.join(_BASE_DIR, "dispense_log.json")


# ── Placeholder image generator ───────────────────────────────────────────────

def _make_placeholder(name: str, path: str) -> None:
    """Create a black-background image with the patient name in white text."""
    if not _CV2:
        print(f"mock_db: cv2 not available — skipping placeholder for {name}")
        return
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        img, name.upper(),
        (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 6,
    )
    cv2.putText(
        img, "(placeholder face)",
        (110, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (160, 160, 160), 2,
    )
    cv2.imwrite(path, img)
    print(f"mock_db: created placeholder → {path}")


# ── Patient records ────────────────────────────────────────────────────────────
# Face images are generated on first run if absent. Pass the path to
# FacialRecognition.enroll() to register each patient.

_PATIENTS: list[dict] = [
    {
        "patient_id":    1,
        "name":          "asshmar",
        "display_name":  "Asshmar",
        "face_image":    os.path.join(_FACES_DIR, "asshmar.jpg"),
        "prescriptions": [
            {
                "medicine_name": "Vitamin A",
                "medicine_code": "VTA01",
                "description":   "yellow capsule",
                "dose":          "800mcg",
                "pill_count":    2,
            }
        ],
    },
    {
        "patient_id":    2,
        "name":          "patient2",
        "display_name":  "Patient 2",
        "face_image":    os.path.join(_FACES_DIR, "patient2.jpg"),
        "prescriptions": [
            {
                "medicine_name": "Vitamin A",
                "medicine_code": "VTA01",
                "description":   "yellow capsule",
                "dose":          "800mcg",
                "pill_count":    1,
            }
        ],
    },
]


def _bootstrap() -> None:
    """Create directories and placeholder face images on first import."""
    os.makedirs(_FACES_DIR, exist_ok=True)
    os.makedirs(_AUDIT_DIR, exist_ok=True)
    for p in _PATIENTS:
        if not os.path.exists(p["face_image"]):
            _make_placeholder(p["name"], p["face_image"])


_bootstrap()


# ── Public API ────────────────────────────────────────────────────────────────

def get_patient_by_name(name: str) -> dict | None:
    """Return the patient dict for the given name, or None if not found."""
    target = name.lower().strip()
    for p in _PATIENTS:
        if p["name"].lower() == target:
            return p
    return None


def get_all_patients() -> list[dict]:
    """Return a list of all patient dicts."""
    return list(_PATIENTS)


def log_dispense(
    patient_id: int,
    medicine_name: str,
    count: int,
    timestamp: str,
    audit_image_path: str | None = None,
) -> None:
    """Append a dispense record to data/dispense_log.json."""
    record = {
        "patient_id":       patient_id,
        "medicine_name":    medicine_name,
        "count":            count,
        "timestamp":        timestamp,
        "audit_image_path": audit_image_path,
    }
    records = get_dispense_log()
    records.append(record)
    with open(_LOG_FILE, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"mock_db: logged dispense → {record}")


def get_dispense_log() -> list[dict]:
    """Return all records from data/dispense_log.json."""
    if not os.path.exists(_LOG_FILE):
        return []
    try:
        with open(_LOG_FILE) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []
