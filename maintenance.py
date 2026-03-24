"""
maintenance.py — PillWheel Maintenance UI
==========================================
Touchscreen interface for testing every hardware component.

Run standalone:
    export DISPLAY=:0
    python3 maintenance.py

Or import and launch from main.py:
    from maintenance import launch_maintenance
    launch_maintenance()
"""

import os
import sys
import time
import threading
import platform
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

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
_ELEC = os.path.join(_ROOT, "electronic")
for _p in (_ROOT, _ELEC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Hardware imports ───────────────────────────────────────────────────────────
from electronic.servo_controller import ServoController
from electronic.tray_sweep import sweep as tray_sweep
from electronic.sound_actuator import SoundActuator, PHRASES

# ── face_tracking — reuse already-loaded module if main.py loaded it first ────
# This avoids opening a second VideoCapture(/dev/video0) which would cause the
# camera to become unavailable to face_tracking.
try:
    if "face_tracking" in sys.modules:
        # main.py already imported it — reuse the same module object
        _ft = sys.modules["face_tracking"]
    elif "electronic.face_tracking" in sys.modules:
        _ft = sys.modules["electronic.face_tracking"]
    else:
        # Running standalone — safe to import fresh
        import electronic.face_tracking as _ft
    _FT_AVAILABLE = _ft is not None
except (ImportError, Exception) as _e:
    _ft = None
    _FT_AVAILABLE = False
    print(f"face_tracking not available ({_e})")

# ── Colours ────────────────────────────────────────────────────────────────────
C_BG      = "#1a1a2e"
C_PANEL   = "#0f0f23"
C_BLUE    = "#4a9eff"
C_GREEN   = "#00c853"
C_RED     = "#ff1744"
C_AMBER   = "#ff9100"
C_PURPLE  = "#9c27b0"
C_WHITE   = "#ffffff"
C_MUTED   = "#888888"
C_BORDER  = "#2a2a4a"

# ── Dimensions ────────────────────────────────────────────────────────────────
W, H = 800, 480


# ══════════════════════════════════════════════════════════════════════════════
#  Helper — consistent button factory
# ══════════════════════════════════════════════════════════════════════════════

def _btn(parent, text, command, bg=C_BLUE, fg=C_WHITE, font=None, **kwargs):
    return tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, font=font,
        relief="flat", cursor="hand2",
        activebackground=bg, activeforeground=fg,
        **kwargs
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MaintenanceApp
# ══════════════════════════════════════════════════════════════════════════════

class MaintenanceApp:

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PillWheel — Maintenance")
        self.root.configure(bg=C_BG)

        self.servo = ServoController()
        self.sound = SoundActuator()

        # camera state shared across camera sub-screen
        self._cap: cv2.VideoCapture | None = None
        self._cam_active = False
        self._cam_label: tk.Label | None = None

        self._init_fonts()
        self._build_ui()
        self.show_screen("home")

    # ── Fonts ──────────────────────────────────────────────────────────────────

    def _init_fonts(self):
        self.f_title  = tkfont.Font(family="Helvetica", size=26, weight="bold")
        self.f_sub    = tkfont.Font(family="Helvetica", size=13)
        self.f_btn    = tkfont.Font(family="Helvetica", size=15, weight="bold")
        self.f_small  = tkfont.Font(family="Helvetica", size=12)
        self.f_servo  = tkfont.Font(family="Helvetica", size=11, weight="bold")

    # ── Screen manager ─────────────────────────────────────────────────────────

    def _build_ui(self):
        self._screens: dict[str, tk.Frame] = {}
        self._container = tk.Frame(self.root, bg=C_BG)
        self._container.pack(fill="both", expand=True)

        for name in ("home", "servo", "camera", "speaker"):
            f = tk.Frame(self._container, bg=C_BG)
            f.place(relx=0, rely=0, relwidth=1, relheight=1)
            self._screens[name] = f

        self._build_home(self._screens["home"])
        self._build_servo(self._screens["servo"])
        self._build_camera(self._screens["camera"])
        self._build_speaker(self._screens["speaker"])

    def show_screen(self, name: str):
        self._screens[name].tkraise()

    # ══════════════════════════════════════════════════════════════════════════
    #  HOME SCREEN
    # ══════════════════════════════════════════════════════════════════════════

    def _build_home(self, f: tk.Frame):
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)
        for r in range(5):
            f.rowconfigure(r, weight=1)

        tk.Label(f, text="Maintenance Mode", bg=C_BG, fg=C_AMBER,
                 font=self.f_title).grid(row=0, column=0, columnspan=2, pady=(30, 4))

        tk.Label(f, text="Select a component to test",
                 bg=C_BG, fg=C_MUTED, font=self.f_sub).grid(row=1, column=0, columnspan=2)

        _btn(f, "⚙  Servos", lambda: self.show_screen("servo"),
             bg=C_BLUE, font=self.f_btn, padx=40, pady=18
             ).grid(row=2, column=0, padx=20, sticky="ew")

        _btn(f, "📷  Camera", lambda: self.show_screen("camera"),
             bg=C_PURPLE, font=self.f_btn, padx=40, pady=18
             ).grid(row=2, column=1, padx=20, sticky="ew")

        _btn(f, "🔊  Speaker", lambda: self.show_screen("speaker"),
             bg=C_GREEN, font=self.f_btn, padx=40, pady=18
             ).grid(row=3, column=0, padx=20, sticky="ew")

        _btn(f, "✖  Quit Program", self._quit,
             bg=C_RED, font=self.f_btn, padx=40, pady=18
             ).grid(row=3, column=1, padx=20, sticky="ew")

        # Status bar
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(f, textvariable=self._status_var, bg=C_PANEL, fg=C_MUTED,
                 font=self.f_small, anchor="w", padx=12
                 ).grid(row=4, column=0, columnspan=2, sticky="ew", ipady=4)

    def _set_status(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        try:
            self._status_var.set(f"{ts}  {msg}")
        except Exception:
            pass

    def _quit(self):
        self._cam_active = False
        time.sleep(0.15)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self.root.destroy()
        sys.exit(0)

    # ══════════════════════════════════════════════════════════════════════════
    #  SERVO SCREEN
    # ══════════════════════════════════════════════════════════════════════════

    def _build_servo(self, f: tk.Frame):
        # Header row
        hdr = tk.Frame(f, bg=C_PANEL)
        hdr.pack(fill="x", padx=0, pady=0)
        _btn(hdr, "← Back", lambda: self.show_screen("home"),
             bg=C_BORDER, font=self.f_small, padx=14, pady=8
             ).pack(side="left", padx=10, pady=8)
        tk.Label(hdr, text="Servo Test", bg=C_PANEL, fg=C_WHITE,
                 font=self.f_btn).pack(side="left", padx=10)

        tk.Label(f, text="Each button fires one full dispense cycle (0° → 180° → 0°)",
                 bg=C_BG, fg=C_MUTED, font=self.f_small).pack(pady=(8, 4))

        # Servo grid — 13 dispensers in 3 rows + tray
        grid_f = tk.Frame(f, bg=C_BG)
        grid_f.pack(fill="both", expand=True, padx=20, pady=4)

        cols = 5
        for i in range(13):
            row, col = divmod(i, cols)
            grid_f.rowconfigure(row, weight=1)
            grid_f.columnconfigure(col, weight=1)
            slot = i + 1
            _btn(grid_f, f"Slot {slot}",
                 lambda idx=i: self._test_dispenser(idx),
                 bg=C_BLUE, font=self.f_servo, padx=4, pady=10
                 ).grid(row=row, column=col, padx=4, pady=4, sticky="ew")

        grid_f.columnconfigure(3, weight=1)
        _btn(grid_f, "Tray Sweep",
             self._test_tray_sweep,
             bg=C_PURPLE, font=self.f_servo, padx=4, pady=10
             ).grid(row=2, column=3, padx=4, pady=4, sticky="ew")

        bot = tk.Frame(f, bg=C_BG)
        bot.pack(fill="x", padx=20, pady=(0, 10))

        self._servo_status = tk.StringVar(value="Idle")
        tk.Label(bot, textvariable=self._servo_status,
                 bg=C_BG, fg=C_AMBER, font=self.f_small).pack(side="left")

        _btn(bot, "Zero All Servos", self._zero_all_servos,
             bg=C_RED, font=self.f_small, padx=16, pady=8
             ).pack(side="right")

    def _set_servo_status(self, msg: str):
        try:
            self._servo_status.set(msg)
        except Exception:
            pass

    def _test_dispenser(self, idx: int):
        self._set_servo_status(f"Running Slot {idx+1}…")
        def run():
            self.servo.rotate_dispenser(idx)
            self.root.after(0, lambda: self._set_servo_status(f"Slot {idx+1} done ✓"))
        threading.Thread(target=run, daemon=True).start()

    def _test_tray_sweep(self):
        self._set_servo_status("Tray sweep running…")
        def run():
            tray_sweep()
            self.root.after(0, lambda: self._set_servo_status("Tray sweep done ✓"))
        threading.Thread(target=run, daemon=True).start()

    def _zero_all_servos(self):
        self._set_servo_status("Zeroing all servos…")
        def run():
            self.servo.cleanup()
            self.root.after(0, lambda: self._set_servo_status("All servos zeroed ✓"))
        threading.Thread(target=run, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    #  CAMERA SCREEN
    # ══════════════════════════════════════════════════════════════════════════

    def _build_camera(self, f: tk.Frame):
        hdr = tk.Frame(f, bg=C_PANEL)
        hdr.pack(fill="x")
        _btn(hdr, "← Back", self._camera_back,
             bg=C_BORDER, font=self.f_small, padx=14, pady=8
             ).pack(side="left", padx=10, pady=8)
        tk.Label(hdr, text="Camera Test", bg=C_PANEL, fg=C_WHITE,
                 font=self.f_btn).pack(side="left", padx=10)

        self._cam_label = tk.Label(f, bg=C_PANEL, text="No feed active",
                                   fg=C_MUTED, font=self.f_sub)
        self._cam_label.pack(fill="both", expand=True, padx=20, pady=8)

        self._cam_status = tk.StringVar(value="Idle")
        tk.Label(f, textvariable=self._cam_status,
                 bg=C_BG, fg=C_AMBER, font=self.f_small).pack(pady=2)

        bot = tk.Frame(f, bg=C_BG)
        bot.pack(fill="x", padx=20, pady=(0, 10))

        _btn(bot, "Live Feed", self._start_live_feed,
             bg=C_BLUE, font=self.f_small, padx=16, pady=8
             ).pack(side="left", padx=6)

        _btn(bot, "Face Tracking", self._start_face_tracking,
             bg=C_PURPLE, font=self.f_small, padx=16, pady=8
             ).pack(side="left", padx=6)

        _btn(bot, "Stop", self._stop_camera,
             bg=C_RED, font=self.f_small, padx=16, pady=8
             ).pack(side="right", padx=6)

    def _camera_back(self):
        self._stop_camera()
        self.show_screen("home")

    def _set_cam_status(self, msg: str):
        try:
            self._cam_status.set(msg)
        except Exception:
            pass

    # ── Live feed ──────────────────────────────────────────────────────────────

    def _start_live_feed(self):
        self._stop_camera()
        self._cam_active = True
        self._set_cam_status("Opening camera…")

        def open_and_loop():
            if _FT_AVAILABLE and _ft is not None:
                # Reuse face_tracking's already-open cap — don't open a new one
                self._set_cam_status("Live feed active (shared camera)")
                while self._cam_active:
                    try:
                        ret, frame = _ft.cap.read()
                    except Exception:
                        break
                    if ret:
                        self._push_frame(frame)
                    time.sleep(0.05)
            else:
                cap = cv2.VideoCapture('/dev/video0')
                if not cap.isOpened():
                    self.root.after(0, lambda: self._set_cam_status("Camera not found"))
                    return
                self._cap = cap
                self._set_cam_status("Live feed active")
                while self._cam_active:
                    ret, frame = cap.read()
                    if ret:
                        self._push_frame(frame)
                    time.sleep(0.05)
                cap.release()
                self._cap = None

            self.root.after(0, lambda: self._set_cam_status("Feed stopped"))

        threading.Thread(target=open_and_loop, daemon=True).start()

    # ── Face tracking ──────────────────────────────────────────────────────────

    def _start_face_tracking(self):
        if not _FT_AVAILABLE:
            self._set_cam_status("face_tracking not available (hardware required)")
            return

        self._stop_camera()
        self._cam_active = True
        self._set_cam_status("Scanning for face…")

        def run():
            face, frame, angle = _ft.scan_for_face(_ft.DEFAULT_ANGLE_CAMERA)

            if frame is not None:
                annotated = frame.copy()
                if face is not None:
                    x, y, w, h = face
                    cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(annotated, "Face detected",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Servo angle: {angle:.0f}deg",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2)
                self._push_frame(annotated)
                self.root.after(0, lambda: self._set_cam_status(
                    f"Face {'detected' if face is not None else 'not found'}"
                    f" — servo at {angle:.0f}°"
                ))
            else:
                self.root.after(0, lambda: self._set_cam_status("No face found in sweep"))

            _ft.set_servo_angle(_ft.DEFAULT_ANGLE_CAMERA)
            self._cam_active = False

        threading.Thread(target=run, daemon=True).start()

    # ── Frame display helper ───────────────────────────────────────────────────

    def _push_frame(self, frame: np.ndarray):
        if not _PIL or self._cam_label is None:
            return
        try:
            h, w = frame.shape[:2]
            scale = min(600 / w, 300 / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (nw, nh))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.root.after(0, lambda p=photo: self._update_cam_label(p))
        except Exception:
            pass

    def _update_cam_label(self, photo):
        try:
            self._cam_label.config(image=photo, text="")
            self._cam_label.image = photo
        except Exception:
            pass

    def _stop_camera(self):
        self._cam_active = False
        time.sleep(0.12)
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._cap = None
        try:
            self._cam_label.config(image="", text="No feed active")
            self._cam_label.image = None
        except Exception:
            pass
        self._set_cam_status("Idle")

    # ══════════════════════════════════════════════════════════════════════════
    #  SPEAKER SCREEN
    # ══════════════════════════════════════════════════════════════════════════

    def _build_speaker(self, f: tk.Frame):
        hdr = tk.Frame(f, bg=C_PANEL)
        hdr.pack(fill="x")
        _btn(hdr, "← Back", lambda: self.show_screen("home"),
             bg=C_BORDER, font=self.f_small, padx=14, pady=8
             ).pack(side="left", padx=10, pady=8)
        tk.Label(hdr, text="Speaker Test", bg=C_PANEL, fg=C_WHITE,
                 font=self.f_btn).pack(side="left", padx=10)

        tk.Label(f, text="Press a button to play that phrase",
                 bg=C_BG, fg=C_MUTED, font=self.f_small).pack(pady=(8, 4))

        scroll_f = tk.Frame(f, bg=C_BG)
        scroll_f.pack(fill="both", expand=True, padx=20, pady=4)

        phrase_map = [
            ("ready",          "Ready for collection",  self.sound.ready_for_collection),
            ("verifying",      "Verifying face",         self.sound.verifying_face),
            ("verified",       "Identity confirmed",     self.sound.verified),
            ("access_denied",  "Access denied",          self.sound.access_denied),
            ("dispensing",     "Dispensing",             self.sound.dispensing),
            ("take_with_food", "Take with food",         self.sound.take_with_food),
            ("collected",      "Collected",              self.sound.collected),
            ("error",          "Error",                  self.sound.error),
            ("missed_dose",    "Missed dose",            self.sound.missed_dose),
            ("low_stock",      "Low stock",              self.sound.low_stock),
            ("no_prescription","No prescription",        self.sound.no_prescription),
            ("count_mismatch", "Count mismatch",         self.sound.count_mismatch),
        ]

        cols = 3
        for i, (key, label, method) in enumerate(phrase_map):
            row, col = divmod(i, cols)
            scroll_f.rowconfigure(row, weight=1)
            scroll_f.columnconfigure(col, weight=1)
            preview = PHRASES.get(key, "")[:48] + ("…" if len(PHRASES.get(key, "")) > 48 else "")
            btn_f = tk.Frame(scroll_f, bg=C_BORDER, padx=2, pady=2)
            btn_f.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            _btn(btn_f, label,
                 lambda m=method, k=key: self._play_phrase(m, k),
                 bg=C_GREEN, font=self.f_servo, padx=6, pady=10
                 ).pack(fill="x")
            tk.Label(btn_f, text=preview, bg=C_BORDER, fg=C_MUTED,
                     font=tkfont.Font(family="Helvetica", size=9),
                     wraplength=180, justify="left"
                     ).pack(fill="x", padx=4, pady=2)

        self._speaker_status = tk.StringVar(value="Idle")
        tk.Label(f, textvariable=self._speaker_status,
                 bg=C_BG, fg=C_AMBER, font=self.f_small).pack(pady=(4, 8))

    def _play_phrase(self, method, key: str):
        self._speaker_status.set(f"Playing: {key}…")
        def run():
            method()
            self.root.after(0, lambda: self._speaker_status.set(f"Done: {key} ✓"))
        threading.Thread(target=run, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
#  Launch helpers
# ══════════════════════════════════════════════════════════════════════════════

def launch_maintenance():
    """Open maintenance UI in a new Tk window (call from main.py)."""
    win = tk.Toplevel()
    win.title("PillWheel — Maintenance")
    win.configure(bg="#1a1a2e")
    machine = platform.machine()
    on_pi = machine.startswith("aarch") or machine.startswith("armv")
    if on_pi:
        win.attributes("-fullscreen", True)
    else:
        win.geometry(f"{W}x{H}")
    MaintenanceApp(win)


if __name__ == "__main__":
    root = tk.Tk()
    machine = platform.machine()
    on_pi = machine.startswith("aarch") or machine.startswith("armv")
    if on_pi:
        root.attributes("-fullscreen", True)
        root.geometry(f"{W}x{H}")
    else:
        root.geometry(f"{W}x{H}")
        root.resizable(True, True)
    MaintenanceApp(root)
    root.mainloop()