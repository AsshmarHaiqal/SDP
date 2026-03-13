"""
main_v3.py
PillWheel — Patient Collection Flow  (v3)
Compact layout for 800×480 Pi display.
Green + pink accessible colour scheme.

Run on Pi:
    export DISPLAY=:0
    sudo -E python3 main_v3.py
"""

import datetime
import os
import subprocess
import threading
import time
import tkinter as tk
import warnings

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import face_recognition as _fr_lib
    _FR_AVAILABLE = True
except ImportError:
    _fr_lib      = None
    _FR_AVAILABLE = False

from electronic.servo_controller import ServoController

# ── Deployment ────────────────────────────────────────────────────────────────
CAMERA_INDEX = 1      # 0 = laptop  |  1 = Pi
FULLSCREEN   = False  # True on Raspberry Pi

# ── Colour scheme ─────────────────────────────────────────────────────────────
#   Shades of green + rose-pink, black text — WCAG AA compliant
BG        = "#f2f7f3"   # light mint white  (background)
HDR_BG    = "#1b3a2d"   # deep forest green (header bars)
PRIMARY   = "#3d7a52"   # sage green        (primary actions)
SUCCESS   = "#27a348"   # bright green      (confirm / success)
DANGER    = "#b83258"   # deep rose-pink    (cancel / danger)
PINK_LT   = "#f9e8ef"   # blush pink        (assistance screen bg)
DISABLED  = "#8aab91"   # muted sage        (disabled)
TEXT      = "#1a1a1a"   # near-black        (body text)
TEXT_LT   = "#ffffff"   # white             (on dark/coloured buttons)
CARD      = "#ffffff"   # white             (card backgrounds)

# ── Fonts  (Helvetica renders cleanly on both Pi and macOS) ──────────────────
F_H1    = ("Helvetica", 34, "bold")
F_H2    = ("Helvetica", 24, "bold")
F_BODY  = ("Helvetica", 17)
F_BTN   = ("Helvetica", 15, "bold")
F_MAINT = ("Helvetica", 13, "bold")
F_SM    = ("Helvetica", 11)

W, H = 800, 480

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = os.path.dirname(os.path.abspath(__file__))
_AUDIT_LOG = os.path.join(_ROOT, "audit_log.txt")


# ── Patients ──────────────────────────────────────────────────────────────────
def _enc(filename):
    path = os.path.join(_ROOT, "faces", filename)
    return path if os.path.exists(path) else None

def _patient_enc_path(patient):
    filename = patient["name"].lower().replace(" ", "_") + ".npy"
    return os.path.join(_ROOT, "faces", filename)

PATIENTS = [
    {"id": 1, "name": "Asshmar",   "encoding": _enc("asshmar.npy")},
    {"id": 2, "name": "Patient 2", "encoding": None},
    {"id": 3, "name": "Patient 3", "encoding": None},
    {"id": 4, "name": "Patient 4", "encoding": None},
    {"id": 5, "name": "Patient 5", "encoding": None},
]


# ── TTS ───────────────────────────────────────────────────────────────────────
def speak(text: str):
    def _say():
        try:
            import platform
            cmd = ["say", text] if platform.system() == "Darwin" else ["espeak-ng", text]
            subprocess.call(cmd)
        except Exception as e:
            warnings.warn(f"TTS unavailable: {e}")
    threading.Thread(target=_say, daemon=True).start()


# ── Shared UI helpers ─────────────────────────────────────────────────────────

def _header(parent, title: str, cancel_cmd=None):
    """
    52-px dark-green header bar with title.
    If cancel_cmd is given, adds a rose ✕ Cancel button on the right.
    Returns (bar, cancel_btn) — cancel_btn is None when not created.
    """
    bar = tk.Frame(parent, bg=HDR_BG, height=52)
    bar.pack(fill="x")
    bar.pack_propagate(False)

    tk.Label(bar, text=title, bg=HDR_BG, fg=TEXT_LT,
             font=F_H2, anchor="w").pack(side="left", padx=14)

    cancel_btn = None
    if cancel_cmd:
        cancel_btn = tk.Button(
            bar, text="✕  Cancel",
            font=F_SM, bg=DANGER, fg=TEXT_LT,
            activebackground=DANGER, activeforeground=TEXT_LT,
            relief="flat", bd=0, padx=12, pady=5,
            command=cancel_cmd)
        cancel_btn.pack(side="right", padx=8, pady=7)

    return bar, cancel_btn


def _btn(parent, text, cmd, bg=PRIMARY, fg=TEXT_LT, font=None, **kw):
    """Standard action button."""
    return tk.Button(
        parent, text=text, command=cmd,
        bg=bg, fg=fg, activebackground=bg, activeforeground=fg,
        font=font or F_BTN, relief="raised", bd=2, **kw)


# ══════════════════════════════════════════════════════════════════════════════
#  SCREEN 1 — MAIN SCREEN
# ══════════════════════════════════════════════════════════════════════════════

class MainScreen(tk.Frame):
    def __init__(self, app):
        super().__init__(app, bg=BG)
        self._app = app

        # Header — Maintenance button lives here, top-right
        bar, _ = _header(self, "💊  PillWheel")
        tk.Button(
            bar, text="Maintenance",
            font=F_SM, bg="#2d5a40", fg=TEXT_LT,
            activebackground="#2d5a40", activeforeground=TEXT_LT,
            relief="flat", bd=0, padx=10, pady=5,
            command=lambda: app.show("MaintenanceScreen"),
        ).pack(side="right", padx=8, pady=7)

        tk.Label(self, text="Select a patient to begin collection",
                 font=F_BODY, fg=TEXT, bg=BG).pack(pady=(10, 6))

        grid = tk.Frame(self, bg=BG)
        grid.pack()

        for i, patient in enumerate(PATIENTS):
            row, col = divmod(i, 3)
            _btn(grid, patient["name"],
                 lambda p=patient: self._select(p),
                 width=14, height=2,
                 ).grid(row=row, column=col, padx=6, pady=5)

    def _select(self, patient):
        if not patient["encoding"]:
            self._popup(patient["name"])
            return
        self._app.current_patient = patient
        self._app.show("CallingScreen")

    def _popup(self, name):
        ov = tk.Frame(self, bg=CARD, relief="solid", bd=1,
                      highlightbackground=PRIMARY, highlightthickness=2)
        ov.place(relx=0.5, rely=0.5, anchor="center", width=440, height=200)
        tk.Label(ov, text="Patient Not Registered",
                 font=F_H2, fg=TEXT, bg=CARD).pack(pady=(18, 6))
        tk.Label(ov, text=f"{name} has no registered face.\n"
                          "Please register in Maintenance.",
                 font=F_BODY, fg=TEXT, bg=CARD, justify="center").pack()
        _btn(ov, "OK", ov.destroy, width=10, height=2).pack(pady=12)


# ═════════════════════════════════���════════════════════════════════════════════
#  SCREEN 2 — MAINTENANCE
# ══════════════════════════════════════════════════════════════════════════════

class MaintenanceScreen(tk.Frame):
    def __init__(self, app):
        super().__init__(app, bg=BG)
        self._app = app

        _header(self, "⚙  Maintenance")

        grid = tk.Frame(self, bg=BG)
        grid.pack(pady=16)

        self._spk_btn = self._mb(grid, "🔊  Test Speaker",  self._test_speaker, 0, 0)
        self._s1_btn  = self._mb(grid, "⚙  Test Servo 1",  self._test_servo1,  0, 1)
        self._s2_btn  = self._mb(grid, "⚙  Test Servo 2",  self._test_servo2,  0, 2)
        self._mb(grid, "📷  Register Faces",
                 lambda: app.show("RegisterFacePage"),   1, 0)
        self._mb(grid, "← Back to Main",
                 lambda: app.show("MainScreen"), 1, 2, bg=DANGER)

        self._status_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self._status_var,
                 font=F_BODY, fg=TEXT, bg=BG).pack(pady=10)

    @staticmethod
    def _mb(parent, text, cmd, row, col, bg=PRIMARY):
        b = _btn(parent, text, cmd, bg=bg, width=16, height=2, font=F_MAINT)
        b.grid(row=row, column=col, padx=8, pady=6)
        return b

    def _set(self, text): self._status_var.set(text)

    def _test_speaker(self):
        self._spk_btn.config(state="disabled", bg=DISABLED)
        self._set("Playing...")
        def _work():
            phrase = "Testing speaker. One, two, three."
            try:
                import platform
                cmd = ["say", phrase] if platform.system() == "Darwin" \
                    else ["espeak-ng", phrase]
                subprocess.call(cmd)
            except Exception:
                pass
            self._app.after(0, lambda: (
                self._set("Speaker OK ✓"),
                self._spk_btn.config(state="normal", bg=PRIMARY),
            ))
        threading.Thread(target=_work, daemon=True).start()

    def _servo_test(self, btn, label, index):
        btn.config(state="disabled", bg=DISABLED)
        self._set(f"Rotating {label}...")
        def _work():
            self._app.servo.rotate_dispenser(index)
            self._app.after(0, lambda: (
                self._set(f"{label} OK ✓"),
                btn.config(state="normal", bg=PRIMARY),
            ))
        threading.Thread(target=_work, daemon=True).start()

    def _test_servo1(self): self._servo_test(self._s1_btn, "Servo 1", 0)
    def _test_servo2(self): self._servo_test(self._s2_btn, "Servo 2", 1)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — CALLING SCREEN
# ══════════════════════════════════════════════════════════════════════════════

class CallingScreen(tk.Frame):
    def __init__(self, app):
        super().__init__(app, bg=BG)
        self._app = app
        _header(self, "💊  Medication Ready", lambda: app.show("MainScreen"))

        self._msg_var = tk.StringVar()
        tk.Label(self, textvariable=self._msg_var,
                 font=F_H2, fg=TEXT, bg=BG,
                 wraplength=680, justify="center").pack(expand=True)

        _btn(self, "Ready to Collect",
             lambda: app.show("FaceVerifyScreen"),
             width=20, height=2).pack(pady=(0, 40))

    def on_show(self):
        name = self._app.current_patient["name"]
        self._msg_var.set(f"{name}, your medication\nis ready for collection.")
        speak(f"{name}, please collect your medication.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — FACE VERIFY SCREEN
# ══════════════════════════════════════════════════════════════════════════════

class FaceVerifyScreen(tk.Frame):
    _FEED_W       = 360
    _FEED_H       = 258
    _MAX_ATTEMPTS = 5

    def __init__(self, app):
        super().__init__(app, bg=BG)
        self._app          = app
        self._stop_feed    = threading.Event()
        self._cancel_widget = None

        _, self._cancel_widget = _header(
            self, "👤  Face Verification", self._cancel)

        tk.Label(self, text="Please look at the camera.",
                 font=F_BODY, fg=TEXT, bg=BG).pack(pady=(8, 4))

        self._feed_lbl = tk.Label(self, bg="#000000")
        self._feed_lbl.pack()

        self._status_var = tk.StringVar(value="")
        self._status_lbl = tk.Label(self, textvariable=self._status_var,
                                    font=F_BODY, fg=TEXT, bg=BG)
        self._status_lbl.pack(pady=6)

        self._return_btn = _btn(
            self, "Return to Main",
            lambda: app.show("MainScreen"),
            bg=DISABLED, width=16, height=2)

    def on_show(self):
        self._stop_feed.set()
        self._stop_feed.clear()
        self.configure(bg=BG)
        self._status_lbl.configure(bg=BG, fg=TEXT)
        self._status_var.set("Scanning...")
        self._return_btn.pack_forget()
        if self._cancel_widget:
            self._cancel_widget.config(state="normal", bg=DANGER)
        speak("Performing facial recognition. Please look at the camera.")
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _camera_loop(self):
        if not _FR_AVAILABLE:
            self._app.after(0, lambda: self._on_result(False))
            return

        patient       = self._app.current_patient
        ref           = np.load(patient["encoding"])
        cap           = cv2.VideoCapture(CAMERA_INDEX)
        verified      = False
        face_attempts = 0

        while not self._stop_feed.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            display   = cv2.resize(frame, (self._FEED_W, self._FEED_H))
            rgb       = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            small     = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
            locations = _fr_lib.face_locations(small)
            encodings = _fr_lib.face_encodings(small, locations)

            for (top, right, bottom, left), enc in zip(locations, encodings):
                top *= 2; right *= 2; bottom *= 2; left *= 2
                matched = _fr_lib.compare_faces([ref], enc, tolerance=0.5)[0]
                if matched:
                    verified = True
                else:
                    face_attempts += 1
                color = (0, 200, 80) if matched else (220, 50, 50)
                cv2.rectangle(rgb, (left, top), (right, bottom), color, 3)

            photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self._app.after(0, lambda p=photo: self._update_feed(p))

            if verified:
                time.sleep(0.8)
                break
            if face_attempts >= self._MAX_ATTEMPTS:
                break
            time.sleep(0.04)

        cap.release()
        if not self._stop_feed.is_set():
            self._app.after(0, lambda: self._on_result(verified))

    def _update_feed(self, photo):
        self._feed_lbl.configure(image=photo)
        self._feed_lbl.image = photo

    def _on_result(self, verified):
        self._feed_lbl.configure(image="")
        self._feed_lbl.image = None

        if verified:
            self._status_var.set("")
            self.configure(bg=SUCCESS)
            self._status_lbl.configure(bg=SUCCESS)
            speak("Identity verified.")
            self.after(1000, lambda: self._app.show("DispensingScreen"))
        else:
            self.configure(bg=DANGER)
            self._status_lbl.configure(bg=DANGER, fg=TEXT_LT)
            self._status_var.set(
                "Verification failed. Please call for assistance.")
            speak("Verification failed. Please call for assistance.")
            if self._cancel_widget:
                self._cancel_widget.config(state="disabled", bg=DISABLED)
            self._return_btn.pack(pady=12)

    def _cancel(self):
        self._stop_feed.set()
        self._app.show("MainScreen")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — DISPENSING SCREEN
# ══════════════════════════════════════════════════════════════════════════════

class DispensingScreen(tk.Frame):
    def __init__(self, app):
        super().__init__(app, bg=BG)
        self._app = app

        _header(self, "⚙  Dispensing Medication")

        tk.Label(self, text="Dispensing your medication.",
                 font=F_H2, fg=TEXT, bg=BG).pack(pady=(44, 6))
        tk.Label(self, text="Please wait.",
                 font=F_BODY, fg=TEXT, bg=BG).pack()

        self._progress_var = tk.StringVar()
        tk.Label(self, textvariable=self._progress_var,
                 font=F_BODY, fg=PRIMARY, bg=BG).pack(pady=14)

    def on_show(self):
        self._progress_var.set("")
        self._app._dispensing = True
        speak("Dispensing your medication.")
        threading.Thread(target=self._dispense, daemon=True).start()

    def _dispense(self):
        for label, index in [("Dispensing 1 of 2...", 0),
                              ("Dispensing 2 of 2...", 1)]:
            self._app.after(0, lambda l=label: self._progress_var.set(l))
            self._app.servo.rotate_dispenser(index)
            time.sleep(1.5)
        self._app.after(0, self._done)

    def _done(self):
        self._app._dispensing = False
        self._app.show("CollectionScreen")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — COLLECTION + REMINDER SCREEN
# ══════════════════════════════════════════════════════════════════════════════

class CollectionScreen(tk.Frame):
    def __init__(self, app):
        super().__init__(app, bg=BG)
        self._app = app

        _header(self, "💊  Collect Your Medication",
                lambda: app.show("MainScreen"))

        tk.Label(self,
                 text="Please collect your medication from the tray.",
                 font=F_BODY, fg=TEXT, bg=BG,
                 wraplength=700, justify="center").pack(pady=(12, 6))

        # Reminder card
        card = tk.Frame(self, bg=CARD,
                        highlightbackground=PRIMARY, highlightthickness=1)
        card.pack(padx=70, pady=4, fill="x")
        for line in ("💊  Take with food or milk",
                     "💧  Take with a full glass of water",
                     "⏰  Take at the same time each day"):
            tk.Label(card, text=line, font=("Helvetica", 15),
                     fg=TEXT, bg=CARD, anchor="w").pack(
                fill="x", padx=14, pady=3)

        btns = tk.Frame(self, bg=BG)
        btns.pack(pady=8)

        _btn(btns, "✓  I have received the correct medication",
             lambda: app.show("SuccessScreen"),
             bg=SUCCESS, width=28, height=2).pack(pady=4)

        _btn(btns, "✗  Something looks wrong",
             lambda: app.show("AssistanceScreen"),
             bg=DANGER, width=28, height=2).pack(pady=4)

    def on_show(self):
        speak("Please collect your medication from the tray.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5a — SUCCESS SCREEN
# ══════════════════════════════════════════════════════════════════════════════

class SuccessScreen(tk.Frame):
    def __init__(self, app):
        super().__init__(app, bg=SUCCESS)
        self._app     = app
        self._msg_var = tk.StringVar()
        tk.Label(self, textvariable=self._msg_var,
                 font=F_H1, fg=TEXT_LT, bg=SUCCESS,
                 justify="center").pack(expand=True)

    def on_show(self):
        patient = self._app.current_patient
        name    = patient["name"] if patient else "Patient"
        self._msg_var.set(f"✓  Thank you, {name}.\nTake care!")
        speak("Thank you. Please take your medication as directed.")
        self.after(3000, lambda: self._app.show("MainScreen"))


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5b — ASSISTANCE SCREEN
# ══════════════════════════════════════════════════════════════════════════════

class AssistanceScreen(tk.Frame):
    def __init__(self, app):
        super().__init__(app, bg=PINK_LT)
        self._app = app

        tk.Label(self, text="⚠  A carer has been notified.\nPlease wait.",
                 font=F_H1, fg=DANGER, bg=PINK_LT,
                 justify="center").pack(expand=True)

        _btn(self, "Return to Main",
             lambda: app.show("MainScreen"),
             bg=HDR_BG, fg=TEXT_LT, width=18, height=2,
             ).pack(pady=(0, 36))

    def on_show(self):
        speak("Please wait. A carer has been notified.")
        self._log_event()

    def _log_event(self):
        patient = self._app.current_patient
        name    = patient["name"] if patient else "Unknown"
        ts      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(_AUDIT_LOG, "a") as f:
            f.write(f"[{ts}] DISCREPANCY — Patient: {name}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  REGISTER FACE PAGE
# ══════════════════════════════════════════════════════════════════════════════

class RegisterFacePage(tk.Frame):
    _PROTECTED_ID = 1   # Asshmar — never cleared from here

    def __init__(self, app):
        super().__init__(app, bg=BG)
        self._app  = app
        self._rows = {}

        _header(self, "📷  Manage Patient Faces")

        rows_frame = tk.Frame(self, bg=BG)
        rows_frame.pack(fill="x", padx=40, pady=8)

        for patient in PATIENTS:
            if patient["id"] == self._PROTECTED_ID:
                continue
            self._build_row(rows_frame, patient)

        self._status_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self._status_var,
                 font=F_BODY, fg=TEXT, bg=BG).pack(pady=6)

        _btn(self, "← Back to Maintenance",
             lambda: app.show("MaintenanceScreen"),
             bg=DANGER, width=22, height=2).pack(pady=4)

    def on_show(self):
        self._status_var.set("")
        for pid, row in self._rows.items():
            patient  = row["patient"]
            has_face = bool(patient["encoding"] and
                            os.path.exists(patient["encoding"]))
            row["status_var"].set("✓ Registered" if has_face else "Not registered")
            row["status_lbl"].config(fg=SUCCESS if has_face else DISABLED)
            row["clear_btn"].config(
                state="normal" if has_face else "disabled",
                bg=DANGER if has_face else DISABLED)

    def _build_row(self, parent, patient):
        pid      = patient["id"]
        has_face = bool(patient["encoding"] and
                        os.path.exists(patient["encoding"]))

        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=4)

        tk.Label(row, text=patient["name"], font=F_BODY,
                 fg=TEXT, bg=BG, width=12, anchor="w").pack(side="left")

        status_var = tk.StringVar(
            value="✓ Registered" if has_face else "Not registered")
        status_lbl = tk.Label(row, textvariable=status_var,
                               font=F_BODY,
                               fg=SUCCESS if has_face else DISABLED,
                               bg=BG, width=14, anchor="w")
        status_lbl.pack(side="left", padx=(6, 10))

        add_btn = _btn(row, "Capture Face",
                       lambda p=pid: self._capture_face(p),
                       width=13, height=2, font=F_MAINT)
        add_btn.pack(side="left", padx=(0, 6))

        clear_btn = _btn(row, "Clear Face",
                         lambda p=pid: self._clear_face(p),
                         bg=DANGER if has_face else DISABLED,
                         width=11, height=2, font=F_MAINT)
        clear_btn.config(state="normal" if has_face else "disabled")
        clear_btn.pack(side="left")

        self._rows[pid] = {
            "patient":    patient,
            "status_var": status_var,
            "status_lbl": status_lbl,
            "add_btn":    add_btn,
            "clear_btn":  clear_btn,
        }

    def _capture_face(self, pid):
        row     = self._rows[pid]
        patient = row["patient"]
        row["add_btn"].config(state="disabled", bg=DISABLED)
        self._status_var.set(
            f"Capturing {patient['name']} — look at the camera...")

        def _work():
            save_path = _patient_enc_path(patient)
            success   = False
            if _FR_AVAILABLE:
                cap = cv2.VideoCapture(CAMERA_INDEX)
                for _ in range(15):
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    encodings = _fr_lib.face_encodings(rgb)
                    if encodings:
                        os.makedirs(os.path.join(_ROOT, "faces"), exist_ok=True)
                        np.save(save_path, encodings[0])
                        patient["encoding"] = save_path
                        success = True
                        break
                cap.release()
            self._app.after(0, lambda: self._on_capture_done(pid, success))

        threading.Thread(target=_work, daemon=True).start()

    def _on_capture_done(self, pid, success):
        row = self._rows[pid]
        row["add_btn"].config(state="normal", bg=PRIMARY)
        if success:
            row["status_var"].set("✓ Registered")
            row["status_lbl"].config(fg=SUCCESS)
            row["clear_btn"].config(state="normal", bg=DANGER)
            self._status_var.set(
                f"✓ Face registered for {row['patient']['name']}.")
        else:
            self._status_var.set(
                "✗ No face detected — check lighting and try again.")

    def _clear_face(self, pid):
        row     = self._rows[pid]
        patient = row["patient"]
        path    = patient["encoding"]
        if path and os.path.exists(path):
            os.remove(path)
        patient["encoding"] = None
        row["status_var"].set("Not registered")
        row["status_lbl"].config(fg=DISABLED)
        row["clear_btn"].config(state="disabled", bg=DISABLED)
        self._status_var.set(f"✓ Face cleared for {patient['name']}.")


# ══════════════════════════════════════════════════════════════════════════════
#  APP ROOT
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    _SCREENS = (
        MainScreen,
        MaintenanceScreen,
        RegisterFacePage,
        CallingScreen,
        FaceVerifyScreen,
        DispensingScreen,
        CollectionScreen,
        SuccessScreen,
        AssistanceScreen,
    )

    def __init__(self):
        super().__init__()
        self.title("PillWheel")
        self.configure(bg=BG)
        self.geometry(f"{W}x{H}")
        self.resizable(False, False)

        if FULLSCREEN:
            self.attributes("-fullscreen", True)

        self.servo           = ServoController()
        self.current_patient = None
        self._dispensing     = False

        self._frames: dict = {}
        for Cls in self._SCREENS:
            frame = Cls(self)
            self._frames[Cls.__name__] = frame
            frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.bind("<Escape>", self._on_escape)
        self.show("MainScreen")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def show(self, name: str):
        frame = self._frames[name]
        if hasattr(frame, "on_show"):
            frame.on_show()
        frame.tkraise()

    def _on_escape(self, _=None):
        if not self._dispensing:
            self.show("MainScreen")

    def _on_close(self):
        self.servo.cleanup()
        self.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
