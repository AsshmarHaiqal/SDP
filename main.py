"""
main.py — PillWheel Touchscreen Application

Orchestrates facial recognition, pill dispensing, and the tkinter UI for a
Raspberry Pi care-home medication dispenser.

Run on Pi:
    export DISPLAY=:0
    sudo -E python3 main.py

Run on laptop (no hardware):
    python3 main.py
"""

import os
import sys
import time
import platform
import threading
from datetime import datetime

import tkinter as tk
from tkinter import font as tkfont

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
    _PIL = True
except ImportError:
    _PIL = False
    print("WARNING: Pillow not installed — pip install Pillow")

# ── Path setup ─────────────────────────────────────────────────────────────────
# Adds project root and electronic/ so that:
#   - "from electronic.xxx import ..." works from root
#   - "from fr_models import ..." works inside facial_recognition.py
#   - "import face_tracking" works (face_tracking.py lives in electronic/)
_ROOT = os.path.dirname(os.path.abspath(__file__))
_ELEC = os.path.join(_ROOT, "electronic")
for _p in (_ROOT, _ELEC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Electronic module imports ──────────────────────────────────────────────────
from electronic.servo_controller import ServoController
from electronic.pill_recogniser import PillRecogniser
from electronic.tray_sweep import sweep as _tray_sweep
from electronic.sound_actuator import SoundActuator
from api_client import APIClient
from maintenance import launch_maintenance

# ── Facial recognition mode ────────────────────────────────────────────────────
# "server" — send frame to server; Claude Vision matches against enrolled photos
# "local"  — download enrolled photos at startup, match on-device using FR model
FR_MODE = os.environ.get("FR_MODE", "server").lower()   # override: export FR_MODE=local

if FR_MODE == "local":
    from electronic.facial_recognition import FacialRecognition

# face_tracking.py opens VideoCapture(0) and ServoKit at module level — only
# available on Pi with hardware. When present it owns the camera and servo 14.
try:
    import face_tracking as _ft
    _FACE_TRACKING = True
    print("face_tracking: loaded — will use _ft.cap and _ft.kit for face scan")
except (ImportError, Exception) as _e:
    _ft = None
    _FACE_TRACKING = False
    print(f"face_tracking not available ({_e}) — using inline fallback")



# ── Colours ────────────────────────────────────────────────────────────────────
C_BG      = "#1a1a2e"
C_PANEL   = "#0f0f23"
C_BLUE    = "#4a9eff"
C_SUCCESS = "#00c853"
C_ERROR   = "#ff1744"
C_WARN    = "#ff9100"
C_WHITE   = "#ffffff"
C_MUTED   = "#888888"

# ── Font sizes (pt) ────────────────────────────────────────────────────────────
F_TITLE  = 36
F_BODY   = 24
F_BTN    = 28
F_STATUS = 14
F_SMALL  = 16

# ── Display ────────────────────────────────────────────────────────────────────
W, H = 800, 480

# Fallback face-scan constants (used when face_tracking is not available)
_SCAN_HOME = 180   # degrees
_SCAN_MIN  =   0
_SCAN_STEP =  10   # larger step → faster sweep (set_servo_angle sleeps 0.5 s/step)


# ══════════════════════════════════════════════════════════════════════════════
#  PillWheelApp
# ══════════════════════════════════════════════════════════════════════════════

class PillWheelApp:
    """
    Touchscreen UI and hardware orchestration for PillWheel.

    Hardware modules used:
        electronic/servo_controller.py  — ServoController  (dispense, camera pan)
        electronic/tray_sweep.py        — sweep()          (funnel pills to cup)
        electronic/pill_recogniser.py   — PillRecogniser   (Claude Vision count)
        electronic/face_tracking.py     — scan_for_face()  (servo + Haar cascade)
        electronic/facial_recognition.py— FacialRecognition(identify resident)
        electronic/sound_actuator.py    — SoundActuator    (TTS feedback)

    All hardware calls run in daemon threads; UI updates are posted back to
    the tkinter main loop via root.after(0, callback).
    """

    SCAN_TIMEOUT   = 15   # seconds before face-scan gives up
    MAX_PILL_RETRY = 3    # attempts to verify each dispensed pill
    COMPLETE_DELAY = 5    # seconds on Complete screen before returning home
    ERROR_DELAY    = 10   # seconds on Error screen before returning home

    def __init__(self, root: tk.Tk) -> None:
        self.root = root

        # ── Hardware ──────────────────────────────────────────────────────────
        # ServoController: rotate_dispenser(slot) for dispensing,
        #                  set_servo_angle(ch, angle) for camera pan (fallback)
        self.servo = ServoController()

        # PillRecogniser: count_pills(frame=<np.ndarray>) → (count, description)
        # Passing a frame avoids opening a second VideoCapture on the Pi.
        self.pill_rec = PillRecogniser()

        self.sound = SoundActuator()

        # APIClient handles all server communication.
        self.api = APIClient()

        # ── Facial recognition ────────────────────────────────────────────────
        # FR_MODE="server"  → frame is sent to server; Claude Vision identifies
        # FR_MODE="local"   → faces synced from server at startup, matched on-Pi
        self.fr: "FacialRecognition | None" = None
        if FR_MODE == "local":
            self._sync_and_enroll()

        # ── Camera ────────────────────────────────────────────────────────────
        # When face_tracking is loaded it owns VideoCapture(0) via _ft.cap.
        # In that case self.cap is never opened to avoid a dual-open conflict.
        self.cap           : cv2.VideoCapture | None = None
        self._cam_lock     = threading.Lock()
        self._latest_frame : np.ndarray | None       = None
        self._feed_active  = False
        self._feed_label   : tk.Label | None         = None

        # ── Session state ─────────────────────────────────────────────────────
        self.current_patient: dict | None = None
        self._stop_flag = threading.Event()

        # Build and show UI
        self._init_fonts()
        self._build_ui()
        self._set_status("System ready")
        self.show_screen("home")

    # ── Local FR startup sync ──────────────────────────────────────────────────

    def _sync_and_enroll(self) -> None:
        """
        Download all enrolled patient face images from the server, save them as
        <patientId>.jpg in data/server_faces/, then enroll each one into the
        local FacialRecognition model.

        Filenames are patientId integers so identify() can later recover the id
        from the matched name string.
        """
        faces_dir = os.path.join(_ROOT, "data", "server_faces")
        print(f"FR_MODE=local — syncing faces from server to {faces_dir}")

        saved = self.api.sync_faces_locally(faces_dir)
        if saved == 0:
            print("WARNING: no face images synced — local FR will find no matches")

        self.fr = FacialRecognition()

        # Enroll each downloaded image (name = str(patientId))
        for fname in os.listdir(faces_dir):
            if not fname.endswith(".jpg"):
                continue
            name = fname.replace(".jpg", "")   # e.g. "42"
            path = os.path.join(faces_dir, fname)
            print(f"Enrolling patient id={name} from {path}")
            self.fr.enroll(name, path)

        print(f"Local FR ready — {len(self.fr.list_enrolled())} patient(s) enrolled")

    # ── Fonts ──────────────────────────────────────────────────────────────────

    def _init_fonts(self) -> None:
        self.f_title  = tkfont.Font(family="Helvetica", size=F_TITLE, weight="bold")
        self.f_body   = tkfont.Font(family="Helvetica", size=F_BODY)
        self.f_btn    = tkfont.Font(family="Helvetica", size=F_BTN,   weight="bold")
        self.f_status = tkfont.Font(family="Helvetica", size=F_STATUS)
        self.f_small  = tkfont.Font(family="Helvetica", size=F_SMALL)

    # ── Status bar ─────────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._status_var.set(f"{ts}   {msg}")

    # ══════════════════════════════════════════════════════════════════════════
    #  UI construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        self._status_var = tk.StringVar()

        outer = tk.Frame(self.root, bg=C_BG)
        outer.pack(fill="both", expand=True)

        tk.Label(
            self.root,
            textvariable=self._status_var,
            bg=C_PANEL, fg=C_MUTED,
            font=self.f_status,
            anchor="w", padx=12, pady=4,
        ).pack(side="bottom", fill="x")

        self._container = tk.Frame(outer, bg=C_BG)
        self._container.pack(fill="both", expand=True)

        self._screens: dict[str, tk.Frame] = {}
        for name in ("home", "scanning", "verified", "dispensing", "complete", "error"):
            frame = tk.Frame(self._container, bg=C_BG)
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)
            self._screens[name] = frame

        self._build_home(self._screens["home"])
        self._build_scanning(self._screens["scanning"])
        self._build_verified(self._screens["verified"])
        self._build_dispensing(self._screens["dispensing"])
        self._build_complete(self._screens["complete"])
        self._build_error(self._screens["error"])

    def show_screen(self, name: str) -> None:
        self._screens[name].tkraise()

    # ── Home ───────────────────────────────────────────────────────────────────

    def _build_home(self, f: tk.Frame) -> None:
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=2)
        f.rowconfigure(1, weight=2)
        f.rowconfigure(2, weight=1)
        f.rowconfigure(3, weight=1)

        tk.Label(f, text="PillWheel", bg=C_BG, fg=C_BLUE,
                 font=self.f_title).grid(row=0, column=0, pady=(30, 0))

        tk.Button(
            f, text="Ready to Collect",
            bg=C_BLUE, fg=C_WHITE, font=self.f_btn,
            relief="flat", cursor="hand2", padx=30, pady=20,
            command=self._start_collection,
        ).grid(row=1, column=0)

        tk.Label(f, text="Press to begin your medication collection",
                 bg=C_BG, fg=C_MUTED, font=self.f_small).grid(row=2, column=0)

        tk.Button(
            f, text="Maintenance",
            bg="#2a2a4a", fg=C_MUTED, font=self.f_status,
            relief="flat", cursor="hand2", padx=14, pady=6,
            command=launch_maintenance,
        ).grid(row=3, column=0, pady=(0, 10))

    # ── Scanning ───────────────────────────────────────────────────────────────

    def _build_scanning(self, f: tk.Frame) -> None:
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=0)
        f.rowconfigure(1, weight=1)
        f.rowconfigure(2, weight=0)
        f.rowconfigure(3, weight=0)

        self._scan_heading = tk.StringVar(value="Scanning for face.")
        tk.Label(f, textvariable=self._scan_heading, bg=C_BG, fg=C_WHITE,
                 font=self.f_title).grid(row=0, column=0, pady=(20, 4))

        self._scan_feed = tk.Label(f, bg=C_PANEL)
        self._scan_feed.grid(row=1, column=0, padx=20, pady=6, sticky="nsew")

        tk.Label(f, text="Hold still and look at the camera",
                 bg=C_BG, fg=C_MUTED, font=self.f_small).grid(row=2, column=0, pady=2)

        tk.Button(f, text="Cancel", bg=C_ERROR, fg=C_WHITE, font=self.f_small,
                  relief="flat", cursor="hand2", padx=20, pady=8,
                  command=self._cancel_to_home).grid(
            row=3, column=0, sticky="e", padx=30, pady=10)

    # ── Verified ───────────────────────────────────────────────────────────────

    def _build_verified(self, f: tk.Frame) -> None:
        f.columnconfigure(0, weight=1)
        for r in range(4):
            f.rowconfigure(r, weight=1)

        self._verified_name = tk.StringVar()
        self._verified_rx   = tk.StringVar()

        tk.Label(f, text="Identity Confirmed", bg=C_BG, fg=C_SUCCESS,
                 font=self.f_title).grid(row=0, column=0, pady=(30, 4))
        tk.Label(f, textvariable=self._verified_name, bg=C_BG, fg=C_WHITE,
                 font=self.f_body).grid(row=1, column=0)
        tk.Label(f, textvariable=self._verified_rx, bg=C_BG, fg=C_BLUE,
                 font=self.f_body).grid(row=2, column=0)
        tk.Label(f, text="Preparing your medication...", bg=C_BG, fg=C_MUTED,
                 font=self.f_small).grid(row=3, column=0)

    # ── Dispensing ─────────────────────────────────────────────────────────────

    def _build_dispensing(self, f: tk.Frame) -> None:
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=0)
        f.rowconfigure(1, weight=1)
        f.rowconfigure(2, weight=0)

        tk.Label(f, text="Dispensing Medication", bg=C_BG, fg=C_WHITE,
                 font=self.f_title).grid(row=0, column=0, pady=(20, 4))

        self._disp_feed = tk.Label(f, bg=C_PANEL)
        self._disp_feed.grid(row=1, column=0, padx=20, pady=6, sticky="nsew")

        self._disp_status_var = tk.StringVar()
        tk.Label(f, textvariable=self._disp_status_var, bg=C_BG, fg=C_WARN,
                 font=self.f_body).grid(row=2, column=0, pady=8)

    # ── Complete ───────────────────────────────────────────────────────────────

    def _build_complete(self, f: tk.Frame) -> None:
        f.columnconfigure(0, weight=1)
        for r in range(4):
            f.rowconfigure(r, weight=1)

        self._complete_name    = tk.StringVar()
        self._complete_details = tk.StringVar()
        self._complete_cd      = tk.StringVar()

        tk.Label(f, text="Medication Dispensed", bg=C_BG, fg=C_SUCCESS,
                 font=self.f_title).grid(row=0, column=0, pady=(30, 4))
        tk.Label(f, textvariable=self._complete_name, bg=C_BG, fg=C_WHITE,
                 font=self.f_body).grid(row=1, column=0)
        tk.Label(f, textvariable=self._complete_details, bg=C_BG, fg=C_BLUE,
                 font=self.f_body).grid(row=2, column=0)
        tk.Label(f, textvariable=self._complete_cd, bg=C_BG, fg=C_MUTED,
                 font=self.f_small).grid(row=3, column=0, pady=4)

    # ── Error ──────────────────────────────────────────────────────────────────

    def _build_error(self, f: tk.Frame) -> None:
        f.columnconfigure(0, weight=1)
        for r in range(4):
            f.rowconfigure(r, weight=1)

        self._error_msg = tk.StringVar()
        self._error_cd  = tk.StringVar()

        tk.Label(f, text="Something Went Wrong", bg=C_BG, fg=C_ERROR,
                 font=self.f_title).grid(row=0, column=0, pady=(30, 4))
        tk.Label(f, textvariable=self._error_msg, bg=C_BG, fg=C_WHITE,
                 font=self.f_body, wraplength=700, justify="center").grid(
            row=1, column=0, padx=30)
        tk.Label(f, text="Please call for assistance", bg=C_BG, fg=C_WARN,
                 font=self.f_body).grid(row=2, column=0)
        tk.Label(f, textvariable=self._error_cd, bg=C_BG, fg=C_MUTED,
                 font=self.f_small).grid(row=3, column=0, pady=4)

    # ══════════════════════════════════════════════════════════════════════════
    #  Camera management
    # ══════════════════════════════════════════════════════════════════════════

    def _open_camera(self) -> None:
        """Open camera. Skipped when face_tracking owns VideoCapture(0)."""
        if _FACE_TRACKING:
            return   # _ft.cap is always open; we read through it
        with self._cam_lock:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)   # TODO: update index if needed
                time.sleep(0.5)

    def _close_camera(self) -> None:
        """Release camera. When face_tracking is active, only stops the feed loop."""
        self._feed_active = False
        if _FACE_TRACKING:
            return   # leave _ft.cap open for next collection
        time.sleep(0.12)
        with self._cam_lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.cap = None

    def _read_frame(self) -> np.ndarray | None:
        """
        Thread-safe single-frame read.
        Uses face_tracking's global cap when available (avoids dual-open on Pi).
        """
        if _FACE_TRACKING:
            try:
                ret, frame = _ft.cap.read()
                return frame if ret else None
            except Exception:
                return None
        with self._cam_lock:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                return frame if ret else None
        return None

    # ── Continuous camera loop ─────────────────────────────────────────────────

    def _start_camera_loop(self) -> None:
        """
        Background thread that fills self._latest_frame at ~20 fps so that the
        face-scan and dispense threads always have a fresh frame without competing
        on cap.read() calls inside long operations.
        """
        self._feed_active = True

        def _loop() -> None:
            while self._feed_active:
                frame = self._read_frame()
                if frame is not None:
                    self._latest_frame = frame
                time.sleep(0.05)

        threading.Thread(target=_loop, daemon=True).start()

    # ── Feed display (tkinter main thread via root.after) ─────────────────────

    def _start_feed(self, label: tk.Label) -> None:
        self._feed_label = label
        self._update_feed()

    def _stop_feed(self) -> None:
        self._feed_label = None

    def _update_feed(self) -> None:
        if self._feed_label is None:
            return
        if _PIL and self._latest_frame is not None:
            try:
                frame = self._latest_frame
                h, w  = frame.shape[:2]
                scale = min(640 / w, 300 / h)
                nw, nh = int(w * scale), int(h * scale)
                resized = cv2.resize(frame, (nw, nh))
                rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                photo   = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                self._feed_label.config(image=photo)
                self._feed_label.image = photo
            except Exception:
                pass
        self.root.after(100, self._update_feed)

    # ══════════════════════════════════════════════════════════════════════════
    #  Animated heading dots
    # ══════════════════════════════════════════════════════════════════════════

    def _start_dots(self) -> None:
        self._dot_n = 0
        self._tick_dots()

    def _tick_dots(self) -> None:
        dots = "." * ((self._dot_n % 3) + 1)
        self._scan_heading.set(f"Scanning for face{dots}")
        self._dot_n += 1
        self._dot_after = self.root.after(500, self._tick_dots)

    def _stop_dots(self) -> None:
        if hasattr(self, "_dot_after"):
            self.root.after_cancel(self._dot_after)

    # ══════════════════════════════════════════════════════════════════════════
    #  Navigation helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _cancel_to_home(self) -> None:
        self._stop_flag.set()
        self._stop_dots()
        self._stop_feed()
        self._close_camera()
        self.show_screen("home")
        self._set_status("Cancelled")

    def _countdown_to_home(self, seconds: int, cd_var: tk.StringVar) -> None:
        def _tick(n: int) -> None:
            if n <= 0:
                self._stop_feed()
                self._close_camera()
                self.current_patient = None
                self.show_screen("home")
                self._set_status("Ready")
                return
            cd_var.set(f"Returning in {n}...")
            self.root.after(1000, lambda: _tick(n - 1))
        _tick(seconds)

    # ══════════════════════════════════════════════════════════════════════════
    #  Dispense flow — Step 1: Start (button press)
    # ══════════════════════════════════════════════════════════════════════════

    def _start_collection(self) -> None:
        self._stop_flag.clear()
        self.current_patient = None
        self._open_camera()
        self._start_camera_loop()
        self.show_screen("scanning")
        self._start_feed(self._scan_feed)
        self._start_dots()
        self._set_status("Scanning for face...")
        self.sound.verifying_face()
        threading.Thread(target=self._face_scan_thread, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    #  Dispense flow — Step 2: Face scan (background thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _face_scan_thread(self) -> None:
        """Dispatch to face_tracking module or inline fallback."""
        if _FACE_TRACKING:
            self._face_scan_via_ft()
        else:
            self._face_scan_inline()

    def _face_scan_via_ft(self) -> None:
        """
        Face scan using face_tracking's servo (channel 14 via _ft.kit) and
        Haar cascade, reading from _ft.cap.

        The camera loop is paused first so scan_for_face() has exclusive access
        to _ft.cap — concurrent reads from two threads cause ret=False which
        breaks the servo sweep immediately.
        After the scan the camera loop is restarted for the dispensing feed.
        """
        # Give exclusive cap access to scan_for_face
        self._feed_active = False
        time.sleep(0.12)   # let the camera loop thread exit

        deadline = time.time() + self.SCAN_TIMEOUT
        locked: np.ndarray | None = None

        while time.time() < deadline:
            if self._stop_flag.is_set():
                return

            face, frame, _angle = _ft.scan_for_face(_ft.DEFAULT_ANGLE_CAMERA)

            if face is not None and frame is not None:
                self._latest_frame = frame.copy()
                locked = frame.copy()
                break

            # Full sweep with no face — pause briefly then retry
            time.sleep(0.3)

        # Restart camera loop for the dispensing live feed
        self._start_camera_loop()

        if locked is not None:
            self.root.after(0, lambda f=locked: self._on_face_detected(f))
        else:
            self.root.after(0, self._on_face_timeout)

    def _face_scan_inline(self) -> None:
        """
        Fallback face scan when face_tracking is unavailable.
        Uses OpenCV Haar cascade + ServoController.set_servo_angle(14, angle).
        """
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        angle     = float(_SCAN_HOME)
        direction = -1
        deadline  = time.time() + self.SCAN_TIMEOUT
        locked: np.ndarray | None = None

        while time.time() < deadline:
            if self._stop_flag.is_set():
                return

            frame = self._latest_frame
            if frame is not None:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                if len(faces) > 0:
                    locked = frame.copy()
                    break

            # Step camera servo via ServoController
            if direction == -1:
                angle = max(float(_SCAN_MIN), angle - _SCAN_STEP)
                self.servo.set_servo_angle(14, angle)   # channel 14 = camera_control
                if angle <= _SCAN_MIN:
                    direction = 1
            else:
                angle = min(float(_SCAN_HOME), angle + _SCAN_STEP)
                self.servo.set_servo_angle(14, angle)
                if angle >= _SCAN_HOME:
                    direction = -1

        if locked is not None:
            self.root.after(0, lambda f=locked: self._on_face_detected(f))
        else:
            self.root.after(0, self._on_face_timeout)

    def _on_face_detected(self, frame: np.ndarray) -> None:
        self._set_status("Face detected — identifying...")
        threading.Thread(
            target=self._identify_thread, args=(frame,), daemon=True
        ).start()

    def _on_face_timeout(self) -> None:
        self._stop_dots()
        self.sound.access_denied()
        self._show_error(
            "No face detected.\nPlease stand in front of the camera and try again."
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  Dispense flow — Step 3: Identity check (background thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _identify_thread(self, frame: np.ndarray) -> None:
        """Identify the resident — runs off the UI thread."""
        if _FACE_TRACKING:
            _ft.set_servo_angle(_ft.DEFAULT_ANGLE_CAMERA)

        if FR_MODE == "local":
            self._identify_local(frame)
        else:
            self._identify_server(frame)

    def _identify_server(self, frame: np.ndarray) -> None:
        """Send frame to server; server runs Claude Vision and returns patient + prescriptions."""
        patient = self.api.identify_patient(frame)
        if patient:
            self.root.after(0, lambda: self._on_identified(patient))
        else:
            self.root.after(0, lambda: self._on_identity_failed(
                "Face not recognised or no medication due.\n"
                "Please see a member of staff."
            ))

    def _identify_local(self, frame: np.ndarray) -> None:
        """
        Match frame against locally-enrolled faces, then fetch prescriptions
        from the server using the matched patientId.

        Names stored during _sync_and_enroll() are string patientIds ("42"),
        so converting the matched name back to int gives us the server patientId.
        """
        if self.fr is None:
            self.root.after(0, lambda: self._on_identity_failed(
                "Local FR not initialised.\nPlease see a member of staff."
            ))
            return

        matched_name = self.fr.identify(frame=frame)

        if not matched_name:
            self.root.after(0, lambda: self._on_identity_failed(
                "Face not recognised.\nPlease see a member of staff."
            ))
            return

        try:
            patient_id = int(matched_name)
        except ValueError:
            self.root.after(0, lambda: self._on_identity_failed(
                "Identity error.\nPlease see a member of staff."
            ))
            return

        patient = self.api.get_patient_prescriptions(patient_id)
        if patient:
            self.root.after(0, lambda: self._on_identified(patient))
        else:
            self.root.after(0, lambda: self._on_identity_failed(
                "No medication due right now.\nPlease see a member of staff."
            ))

    def _on_identity_failed(self, reason: str) -> None:
        self._stop_dots()
        self.sound.access_denied()
        self._show_error(reason)

    def _on_identified(self, patient: dict) -> None:
        self._stop_dots()
        self._stop_feed()
        self.current_patient = patient
        rx         = patient["prescriptions"][0]
        first_name = patient["display_name"].split()[0]

        self.sound.verified()
        self.sound.speak(f"Welcome, {first_name}")

        self._verified_name.set(f"Welcome, {patient['display_name']}")
        self._verified_rx.set(f"{rx['medicine_name']}  ×  {rx['pill_count']}")
        self.show_screen("verified")
        self._set_status(f"Identified: {patient['display_name']}")
        self.root.after(2000, self._start_dispensing)

    # ══════════════════════════════════════════════════════════════════════════
    #  Dispense flow — Step 4: Dispensing (background thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _start_dispensing(self) -> None:
        if self._stop_flag.is_set() or self.current_patient is None:
            return
        self._start_feed(self._disp_feed)
        self.show_screen("dispensing")
        self.sound.dispensing()
        self._disp_set("Checking tray is empty...")
        threading.Thread(target=self._dispense_thread, daemon=True).start()

    def _disp_set(self, msg: str) -> None:
        """Thread-safe dispensing status label update."""
        self.root.after(0, lambda m=msg: self._disp_status_var.set(m))

    def _dispense_thread(self) -> None:
        patient  = self.current_patient
        rx       = patient["prescriptions"][0]
        expected = rx["pill_count"]
        med      = rx["medicine_name"]
        pid      = patient["patient_id"]

        # medicine_number is 1-based (1-13); servo channel is 0-based (0-12)
        servo_slot = rx.get("medicine_number", 1) - 1

        # ── 1. Wait until tray is clear ────────────────────────────────────────
        while not self._stop_flag.is_set():
            frame = self._latest_frame
            if frame is not None:
                count, _desc = self.pill_rec.count_pills(frame=frame)
                if count == 0:
                    break
            self._disp_set("Please clear the tray before collecting")
            self.sound.speak("Please clear the tray before collecting")
            time.sleep(3)

        if self._stop_flag.is_set():
            return

        self._disp_set("Tray clear. Starting dispense.")
        time.sleep(1)

        # ── 2. Dispense one pill at a time ─────────────────────────────────────
        audit_frame: np.ndarray | None = None

        for i in range(expected):
            if self._stop_flag.is_set():
                return

            self._disp_set(f"Dispensing pill {i+1} of {expected}...")
            self.servo.rotate_dispenser(servo_slot)
            time.sleep(1.5)   # let pill settle on tray

            verified = False
            for attempt in range(self.MAX_PILL_RETRY):
                frame = self._latest_frame
                if frame is None:
                    time.sleep(0.3)
                    continue
                audit_frame        = frame.copy()
                count, description = self.pill_rec.count_pills(frame=frame)

                if count == i + 1:
                    self._disp_set(
                        f"Pill {i+1} of {expected} detected ✓  ({description})"
                    )
                    verified = True
                    break

                self._disp_set(
                    f"Expected {i+1}, detected {count} "
                    f"— retrying ({attempt+1}/{self.MAX_PILL_RETRY})..."
                )
                time.sleep(1.0)

            if not verified:
                self.sound.error()
                self.api.log_dispense_result(
                    patient_id      = pid,
                    prescription_id = rx["prescription_id"],
                    scheduled_time  = rx.get("scheduled_time", ""),
                    status          = "FAILED",
                    failure_reason  = f"Pill {i+1} not confirmed after {self.MAX_PILL_RETRY} attempts",
                )
                idx = i + 1
                self.root.after(
                    0,
                    lambda i=idx: self._show_error(
                        f"Pill {i} not confirmed after {self.MAX_PILL_RETRY} attempts.\n"
                        "Please call for assistance."
                    ),
                )
                return

        # ── 3. Save audit image ────────────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audit_dir = os.path.join(_ROOT, "data", "audit")
        os.makedirs(audit_dir, exist_ok=True)
        audit_path: str | None = None
        if audit_frame is not None:
            audit_path = os.path.join(audit_dir, f"{pid}_{timestamp}.jpg")
            cv2.imwrite(audit_path, audit_frame)

        # ── 4. Sweep tray ──────────────────────────────────────────────────────
        # tray_sweep.sweep() drives channel 15 (tray_tilt) via its own ServoKit
        self._disp_set("Dispensing complete. Collecting medication...")
        _tray_sweep()

        first_name = patient["display_name"].split()[0]
        self.sound.collected()
        self.sound.speak(f"Have a lovely day, {first_name}")

        # ── 5. Log to server and advance to Complete screen ───────────────────
        self.api.log_dispense_result(
            patient_id      = pid,
            prescription_id = rx["prescription_id"],
            scheduled_time  = rx.get("scheduled_time", ""),
            status          = "TAKEN",
        )
        self.api.reduce_stock(med, expected)
        self.root.after(
            0, lambda: self._on_dispense_complete(patient, rx, timestamp)
        )

    def _on_dispense_complete(
        self, patient: dict, rx: dict, timestamp: str
    ) -> None:
        self._stop_feed()
        name = patient["display_name"]
        ts   = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%H:%M  %d/%m/%Y")
        self._complete_name.set(name)
        self._complete_details.set(
            f"{rx['medicine_name']}  ×  {rx['pill_count']}  —  {ts}"
        )
        self._complete_cd.set("")
        self.show_screen("complete")
        self._set_status(f"Dispense complete for {name}")
        self._countdown_to_home(self.COMPLETE_DELAY, self._complete_cd)

    # ══════════════════════════════════════════════════════════════════════════
    #  Error screen
    # ══════════════════════════════════════════════════════════════════════════

    def _show_error(self, message: str) -> None:
        self._stop_feed()
        self._error_msg.set(message)
        self._error_cd.set("")
        self.show_screen("error")
        self._set_status(f"Error: {message.splitlines()[0]}")
        self._countdown_to_home(self.ERROR_DELAY, self._error_cd)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()

    machine = platform.machine()
    on_pi   = machine.startswith("aarch") or machine.startswith("armv")

    if on_pi:
        root.attributes("-fullscreen", True)
        root.geometry(f"{W}x{H}")
    else:
        root.geometry(f"{W}x{H}")
        root.resizable(True, True)

    app = PillWheelApp(root)
    root.mainloop()