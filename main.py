"""
main.py — PillWheel Touchscreen Application

Thin launcher that wires hardware modules into the refactored ui/ package.

    old (asshmarhaiqal):  monolithic PillWheelApp with everything inline
    new (integrated):     delegates to ui.pillwheel_app.PillWheelApp

Run on Pi:
    export DISPLAY=:0
    python3 main.py

Run on laptop (no hardware):
    python3 main.py
"""

import os
import sys
import platform

import tkinter as tk

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
_ELEC = os.path.join(_ROOT, "electronic")
for _p in (_ROOT, _ELEC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Electronic module imports ──────────────────────────────────────────────────
from electronic.servo_controller import ServoController
from electronic.pill_recogniser  import PillRecogniser
from electronic.tray_sweep       import sweep as _tray_sweep
from electronic.sound_actuator   import SoundActuator
from api_client                  import APIClient

# ── Facial recognition mode ────────────────────────────────────────────────────
FR_MODE = os.environ.get("FR_MODE", "server").lower()

if FR_MODE == "local":
    from electronic.facial_recognition import FacialRecognition

# ── face_tracking — MUST be imported before maintenance ───────────────────────
# face_tracking.py opens VideoCapture at module level and owns the camera.
# maintenance.py checks sys.modules for it — so it must be loaded first.
try:
    import face_tracking as _ft
    _FACE_TRACKING = True
    print("face_tracking: loaded ✔")
except (ImportError, Exception) as _e:
    _ft = None
    _FACE_TRACKING = False
    print(f"face_tracking not available ({_e}) — using inline fallback")

# ── Maintenance — imported AFTER face_tracking so sys.modules has it ──────────
from maintenance import launch_maintenance

# ── Display ────────────────────────────────────────────────────────────────────
W, H = 800, 480

# ── UI package (winterscone refactored classes) ───────────────────────────────
from ui.pillwheel_app import PillWheelApp


# ══════════════════════════════════════════════════════════════════════════════
#  Local FR enrolment helper
# ══════════════════════════════════════════════════════════════════════════════

def _sync_and_enroll(api: APIClient) -> "FacialRecognition | None":
    """
    Download patient face images from the server, enrol them locally,
    and return a ready FacialRecognition instance.
    """
    faces_dir = os.path.join(_ROOT, "data", "server_faces")
    os.makedirs(faces_dir, exist_ok=True)
    print(f"FR_MODE=local — syncing faces from server to {faces_dir}")

    saved = api.sync_faces_locally(faces_dir)
    if saved == 0:
        print("WARNING: no face images synced — local FR will find no matches")

    fr = FacialRecognition()

    for fname in os.listdir(faces_dir):
        if not fname.endswith(".jpg"):
            continue
        name = fname.replace(".jpg", "")
        path = os.path.join(faces_dir, fname)
        print(f"Enrolling patient id={name} from {path}")
        fr.enroll(name, path)

    print(f"Local FR ready — {len(fr.list_enrolled())} patient(s) enrolled")
    return fr


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    root.title("PillWheel")

    machine = platform.machine()
    on_pi   = machine.startswith("aarch") or machine.startswith("armv")

    if on_pi:
        root.attributes("-fullscreen", True)
        root.geometry(f"{W}x{H}")
    else:
        root.geometry(f"{W}x{H}")
        root.resizable(True, True)

    # ── Instantiate hardware ──────────────────────────────────────────────────
    servo    = ServoController()
    pill_rec = PillRecogniser()
    sound    = SoundActuator()
    api      = APIClient()

    # ── Local FR (optional) ───────────────────────────────────────────────────
    fr = None
    if FR_MODE == "local":
        fr = _sync_and_enroll(api)

    # ── Build app ─────────────────────────────────────────────────────────────
    app = PillWheelApp(
        root           = root,
        servo          = servo,
        pill_rec       = pill_rec,
        tray_sweep     = _tray_sweep,
        sound          = sound,
        api_client     = api,
        ft             = _ft if _FACE_TRACKING else None,
        fr             = fr,
        fr_mode        = FR_MODE,
        audit_dir      = os.path.join(_ROOT, "data", "audit"),
        on_maintenance = launch_maintenance,
    )

    try:
        root.mainloop()
    finally:
        if _FACE_TRACKING and _ft is not None:
            try:
                if _ft.cap.isOpened():
                    _ft.cap.release()
                    print("Camera released.")
            except Exception:
                pass