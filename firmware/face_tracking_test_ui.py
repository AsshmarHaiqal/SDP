"""
Firmware/face_tracking_test_ui.py
Touchscreen test UI for face_tracking.py

Screens:
    HOME      — "Track Face" / "Close Program"
    TRACKING  — live camera feed while servo scans; "Cancel" aborts
    RESULT    — shows captured snapshot; "Back" discards + returns home
                                          "Close Program" exits

Usage:
    export DISPLAY=:0
    python3 firmware/face_tracking_test_ui.py
"""

import sys
import os
import threading
import time

import cv2
import numpy as np
import tkinter as tk
from tkinter import font as tkfont

try:
    from PIL import Image, ImageTk
    _PIL = True
except ImportError:
    _PIL = False
    print("WARNING: pip install Pillow  — live feed disabled")

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ELECTRONIC = os.path.join(_ROOT, "electronic")   # ← add this

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _ELECTRONIC not in sys.path:
    sys.path.insert(0, _ELECTRONIC)

# face_tracking opens ServoKit + VideoCapture at import time
try:
    import face_tracking as _ft
    _FT_OK = True
    print("face_tracking loaded ✔")
except Exception as _e:
    _ft = None
    _FT_OK = False
    print(f"face_tracking unavailable ({_e}) — simulation mode")

# ── Dimensions ────────────────────────────────────────────────────────────────
W, H = 800, 480

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG      = "#0d1117"
C_PANEL   = "#161b22"
C_BORDER  = "#30363d"
C_BLUE    = "#388bfd"
C_GREEN   = "#238636"
C_RED     = "#b91c1c"
C_WARN    = "#d97706"
C_FG      = "#e6edf3"
C_MUTED   = "#8b949e"
C_SUCCESS = "#3fb950"


class FaceTrackingTestUI:

    SCAN_TIMEOUT = 20   # seconds before giving up

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("PillWheel — Face Tracking Test")
        root.geometry(f"{W}x{H}")
        root.configure(bg=C_BG)
        root.resizable(True, True)
        root.bind("<Escape>", lambda e: self._go_home())

        self._init_fonts()

        # State
        self._stop_flag       = threading.Event()
        self._latest_frame    : np.ndarray | None = None
        self._captured_frame  : np.ndarray | None = None
        self._feed_active     = False
        self._feed_label      : tk.Label | None   = None

        self._build_all_screens()
        self._show("home")

    # ── Fonts ─────────────────────────────────────────────────────────────────

    def _init_fonts(self):
        self.f_title  = tkfont.Font(family="Courier", size=22, weight="bold")
        self.f_sub    = tkfont.Font(family="Courier", size=12)
        self.f_btn    = tkfont.Font(family="Courier", size=15, weight="bold")
        self.f_small  = tkfont.Font(family="Courier", size=11)
        self.f_status = tkfont.Font(family="Courier", size=10)

    # ── Screen management ─────────────────────────────────────────────────────

    def _build_all_screens(self):
        self._screens: dict[str, tk.Frame] = {}
        for name in ("home", "tracking", "result"):
            f = tk.Frame(self.root, bg=C_BG)
            f.place(relx=0, rely=0, relwidth=1, relheight=1)
            self._screens[name] = f

        self._build_home(self._screens["home"])
        self._build_tracking(self._screens["tracking"])
        self._build_result(self._screens["result"])

        # Shared status bar (always on top)
        self._status_var = tk.StringVar(value="")
        tk.Label(
            self.root, textvariable=self._status_var,
            font=self.f_status, bg=C_PANEL, fg=C_MUTED,
            anchor="w", padx=12, pady=4,
        ).place(relx=0, rely=1.0, anchor="sw", relwidth=1)

    def _show(self, name: str):
        self._screens[name].tkraise()

    def _set_status(self, msg: str):
        self._status_var.set(f"  {msg}")

    # ── HOME screen ───────────────────────────────────────────────────────────

    def _build_home(self, f: tk.Frame):
        f.columnconfigure(0, weight=1)
        for r in range(5):
            f.rowconfigure(r, weight=1)

        tk.Label(f, text="FACE TRACKING TEST",
                 font=self.f_title, bg=C_BG, fg=C_FG).grid(
            row=0, column=0, pady=(30, 0))

        hw_ok   = _FT_OK
        hw_text = "Hardware: ✔ ServoKit + camera" if hw_ok else "Hardware: ⚠  simulation mode"
        tk.Label(f, text=hw_text, font=self.f_sub, bg=C_BG,
                 fg=C_SUCCESS if hw_ok else C_WARN).grid(row=1, column=0)

        tk.Button(
            f, text="▶  TRACK FACE",
            font=self.f_btn, bg=C_BLUE, fg=C_FG,
            activebackground="#1d6fd4", relief="flat", cursor="hand2",
            padx=40, pady=18,
            command=self._start_tracking,
        ).grid(row=2, column=0)

        tk.Button(
            f, text="✕  CLOSE PROGRAM",
            font=self.f_btn, bg=C_RED, fg=C_FG,
            activebackground="#991b1b", relief="flat", cursor="hand2",
            padx=40, pady=14,
            command=self._close_program,
        ).grid(row=3, column=0)

        tk.Label(f, text="ESC → home  |  live camera feed requires Pillow",
                 font=self.f_small, bg=C_BG, fg=C_MUTED).grid(row=4, column=0, pady=(0, 16))

    # ── TRACKING screen ───────────────────────────────────────────────────────

    def _build_tracking(self, f: tk.Frame):
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=0)
        f.rowconfigure(1, weight=1)
        f.rowconfigure(2, weight=0)
        f.rowconfigure(3, weight=0)

        self._track_heading = tk.StringVar(value="Scanning for face…")
        tk.Label(f, textvariable=self._track_heading,
                 font=self.f_title, bg=C_BG, fg=C_FG).grid(
            row=0, column=0, pady=(16, 4))

        # Live feed label
        self._live_label = tk.Label(f, bg=C_PANEL,
                                    highlightbackground=C_BORDER, highlightthickness=1)
        self._live_label.grid(row=1, column=0, padx=20, pady=4, sticky="nsew")

        self._track_sub = tk.StringVar(value="Servo sweeping — face will be captured when centred")
        tk.Label(f, textvariable=self._track_sub,
                 font=self.f_small, bg=C_BG, fg=C_MUTED).grid(row=2, column=0, pady=2)

        tk.Button(
            f, text="✕  CANCEL",
            font=self.f_btn, bg=C_RED, fg=C_FG,
            activebackground="#991b1b", relief="flat", cursor="hand2",
            padx=30, pady=10,
            command=self._cancel_tracking,
        ).grid(row=3, column=0, sticky="e", padx=30, pady=(0, 28))

    # ── RESULT screen ─────────────────────────────────────────────────────────

    def _build_result(self, f: tk.Frame):
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(0, weight=0)
        f.rowconfigure(1, weight=1)
        f.rowconfigure(2, weight=0)

        self._result_heading = tk.StringVar(value="Face Captured")
        tk.Label(f, textvariable=self._result_heading,
                 font=self.f_title, bg=C_BG, fg=C_SUCCESS).grid(
            row=0, column=0, columnspan=2, pady=(16, 4))

        self._snap_label = tk.Label(f, bg=C_PANEL,
                                    highlightbackground=C_BORDER, highlightthickness=1)
        self._snap_label.grid(row=1, column=0, columnspan=2,
                              padx=20, pady=4, sticky="nsew")

        btn_frame = tk.Frame(f, bg=C_BG)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(0, 28))

        tk.Button(
            btn_frame, text="◀  BACK",
            font=self.f_btn, bg=C_GREEN, fg=C_FG,
            activebackground="#166534", relief="flat", cursor="hand2",
            padx=30, pady=14,
            command=self._go_home,
        ).pack(side="left", padx=20)

        tk.Button(
            btn_frame, text="✕  CLOSE PROGRAM",
            font=self.f_btn, bg=C_RED, fg=C_FG,
            activebackground="#991b1b", relief="flat", cursor="hand2",
            padx=30, pady=14,
            command=self._close_program,
        ).pack(side="left", padx=20)

    # ── Tracking flow ─────────────────────────────────────────────────────────

    def _start_tracking(self):
        self._stop_flag.clear()
        self._captured_frame = None
        self._set_status("Initialising camera…")
        self._track_heading.set("Scanning for face…")
        self._track_sub.set("Servo sweeping — face will be captured when centred")
        self._show("tracking")

        # Start camera loop for live feed
        self._feed_active = True
        self._feed_label  = self._live_label
        threading.Thread(target=self._camera_loop, daemon=True).start()
        self._update_feed()

        # Start tracking thread
        threading.Thread(target=self._tracking_thread, daemon=True).start()
        self._set_status("Scanning…")

    def _camera_loop(self):
        """Continuously grab frames into self._latest_frame."""
        if _FT_OK:
            src = _ft.cap
            while self._feed_active:
                ret, frame = src.read()
                if ret:
                    self._latest_frame = frame
                time.sleep(0.04)
        else:
            # Simulation: blank grey frame with text
            while self._feed_active:
                img = np.full((240, 320, 3), 40, dtype=np.uint8)
                cv2.putText(img, "SIMULATION MODE", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 150, 255), 2)
                self._latest_frame = img
                time.sleep(0.1)

    def _update_feed(self):
        """Push latest frame to the active feed label (main thread, ~20 fps)."""
        if not self._feed_active or self._feed_label is None:
            return
        if _PIL and self._latest_frame is not None:
            try:
                frame = self._latest_frame
                h, w  = frame.shape[:2]
                scale = min(640 / w, 300 / h)
                nw, nh = int(w * scale), int(h * scale)
                rgb    = cv2.cvtColor(cv2.resize(frame, (nw, nh)), cv2.COLOR_BGR2RGB)
                photo  = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                self._feed_label.config(image=photo)
                self._feed_label.image = photo
            except Exception:
                pass
        self.root.after(50, self._update_feed)

    def _tracking_thread(self):
        """
        Run face_tracking logic in background.
        Uses _ft.scan_for_face() + _ft.cap when hardware is available,
        otherwise simulates a 3-second delay then pretends a face was found.
        """
        if _FT_OK:
            self._run_real_tracking()
        else:
            self._run_simulated_tracking()

    def _run_real_tracking(self):
        # Give camera loop exclusive access: pause feed briefly
        self._feed_active = False
        time.sleep(0.15)

        deadline = time.time() + self.SCAN_TIMEOUT
        captured: np.ndarray | None = None

        current_angle = _ft.set_servo_angle(_ft.DEFAULT_ANGLE_CAMERA)
        time.sleep(0.5)

        while time.time() < deadline:
            if self._stop_flag.is_set():
                _ft.set_servo_angle(_ft.DEFAULT_ANGLE_CAMERA)
                return

            face, frame, current_angle = _ft.scan_for_face(current_angle)

            if face is None:
                # Full sweep found nothing — restart from home
                self.root.after(0, lambda: self._track_sub.set(
                    "No face found in sweep — restarting…"))
                current_angle = _ft.set_servo_angle(_ft.DEFAULT_ANGLE_CAMERA)
                time.sleep(0.3)
                continue

            # Face found — now track until centred
            self.root.after(0, lambda: self._track_heading.set("Face detected — centring…"))
            self._latest_frame = frame.copy()
            self._feed_active  = True
            threading.Thread(target=self._camera_loop, daemon=True).start()
            self.root.after(0, self._update_feed)

            while time.time() < deadline:
                if self._stop_flag.is_set():
                    _ft.set_servo_angle(_ft.DEFAULT_ANGLE_CAMERA)
                    return

                ret, frame = _ft.cap.read()
                if not ret:
                    break

                fh, fw = frame.shape[:2]
                center_y = fh // 2
                gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces    = _ft.face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) == 0:
                    # Lost face — re-scan
                    self._feed_active = False
                    time.sleep(0.12)
                    face, frame, current_angle = _ft.scan_for_face(current_angle)
                    self._feed_active = True
                    if face is None:
                        break
                    continue

                x, y, w, h   = max(faces, key=lambda f: f[2] * f[3])
                face_center_y = y + h // 2
                diff          = face_center_y - center_y

                # Draw overlay
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.line(frame, (0, center_y), (fw, center_y), (255, 0, 0), 1)
                cv2.putText(frame, f"Servo: {current_angle:.0f} deg", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"diff: {diff:+d}px", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                self._latest_frame = frame.copy()

                if abs(diff) <= _ft.FRAME_CENTER_TOLERANCE:
                    captured = frame.copy()
                    break

                if diff > _ft.FRAME_CENTER_TOLERANCE:
                    current_angle = _ft.set_servo_angle(current_angle + _ft.TRACK_STEP)
                else:
                    current_angle = _ft.set_servo_angle(current_angle - _ft.TRACK_STEP)

                time.sleep(0.1)

            if captured is not None:
                break

        _ft.set_servo_angle(_ft.DEFAULT_ANGLE_CAMERA)
        self._feed_active = False

        if captured is not None and not self._stop_flag.is_set():
            self.root.after(0, lambda f=captured: self._on_captured(f))
        elif not self._stop_flag.is_set():
            self.root.after(0, self._on_timeout)

    def _run_simulated_tracking(self):
        """Pretend to scan for 3 s then produce a blank 'captured' frame."""
        for i in range(6):
            if self._stop_flag.is_set():
                return
            self.root.after(0, lambda i=i: self._track_sub.set(
                f"[simulation] scanning… step {i+1}/6"))
            time.sleep(0.5)

        if self._stop_flag.is_set():
            return

        # Fake captured frame
        img = np.full((240, 320, 3), 30, dtype=np.uint8)
        cv2.putText(img, "SIMULATED CAPTURE", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 2)
        self._feed_active = False
        self.root.after(0, lambda: self._on_captured(img))

    # ── Outcome callbacks (main thread) ───────────────────────────────────────

    def _on_captured(self, frame: np.ndarray):
        self._feed_active  = False
        self._feed_label   = None
        self._captured_frame = frame   # held in memory only — never written to disk here

        # Display snapshot on result screen
        self._show_snapshot(frame)
        self._result_heading.set("✔  Face Captured")
        self._show("result")
        self._set_status("Capture complete — press Back to discard or Close to exit.")

    def _on_timeout(self):
        self._feed_active = False
        self._feed_label  = None
        self._track_heading.set("No face found")
        self._track_sub.set("Timed out — tap Cancel to return home")
        self._set_status("Scan timed out.")

    def _cancel_tracking(self):
        self._stop_flag.set()
        self._feed_active = False
        self._feed_label  = None
        self._captured_frame = None
        self._go_home()

    def _go_home(self):
        """Discard any captured frame and return to home screen."""
        self._stop_flag.set()
        self._feed_active    = False
        self._feed_label     = None
        self._captured_frame = None   # explicitly discard
        self._latest_frame   = None
        self._show("home")
        self._set_status("Ready.")

    def _close_program(self):
        if _FT_OK:
            try:
                _ft.set_servo_angle(_ft.DEFAULT_ANGLE_CAMERA)
                _ft.cap.release()
            except Exception:
                pass
        self.root.quit()

    # ── Snapshot helper ───────────────────────────────────────────────────────

    def _show_snapshot(self, frame: np.ndarray):
        if not _PIL:
            return
        try:
            h, w   = frame.shape[:2]
            scale  = min(640 / w, 300 / h)
            nw, nh = int(w * scale), int(h * scale)
            rgb    = cv2.cvtColor(cv2.resize(frame, (nw, nh)), cv2.COLOR_BGR2RGB)
            photo  = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self._snap_label.config(image=photo)
            self._snap_label.image = photo
        except Exception:
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app  = FaceTrackingTestUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app._close_program()