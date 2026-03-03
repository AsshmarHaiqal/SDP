"""
main_v2.py
PillWheel — Patient Collection Flow
Raspberry Pi touchscreen UI (800×480, fullscreen)

Run on Pi:
    export DISPLAY=:0
    sudo -E python3 main_v2.py
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

try:
    import face_recognition as _fr_lib
    _FR_AVAILABLE = True
except ImportError:
    _fr_lib      = None
    _FR_AVAILABLE = False

from electronic.servo_controller import ServoController

# ── Deployment ────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0       # 0 = laptop built-in  |  1 = Pi camera
FULLSCREEN   = False   # True on Raspberry Pi

# ── Colours ───────────────────────────────────────────────────────────────────
BG       = "#1a1a2e"
PRIMARY  = "#00b4d8"
SUCCESS  = "#2dc653"
DANGER   = "#e63946"
DISABLED = "#6c757d"
TEXT     = "#ffffff"

# ── Fonts ─────────────────────────────────────────────────────────────────────
F_H1    = ("Arial", 42, "bold")
F_H2    = ("Arial", 32, "bold")
F_BODY  = ("Arial", 22)
F_BTN   = ("Arial", 22, "bold")
F_MAINT = ("Arial", 18, "bold")
F_SM    = ("Arial", 14)

W, H = 800, 480

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = os.path.dirname(os.path.abspath(__file__))
_AUDIT_LOG = os.path.join(_ROOT, "audit_log.txt")


# ── Patients ──────────────────────────────────────────────────────────────────
def _enc(filename: str):
    """Return absolute path to a face encoding if it exists, else None."""
    path = os.path.join(_ROOT, "faces", filename)
    return path if os.path.exists(path) else None


PATIENTS = [
    {"id": 1, "name": "Asshmar",   "encoding": _enc("asshmar.npy")},
    {"id": 2, "name": "Patient 2", "encoding": None},
    {"id": 3, "name": "Patient 3", "encoding": None},
    {"id": 4, "name": "Patient 4", "encoding": None},
    {"id": 5, "name": "Patient 5", "encoding": None},
]


# ── TTS ───────────────────────────────────────────────────────────────────────
def speak(text: str):
    """Non-blocking TTS. Tries pyttsx3 → espeak → silent with warning."""
    def _say():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 120)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception:
            try:
                subprocess.call(["espeak", text])
            except Exception:
                warnings.warn(f"TTS unavailable: {text}")

    threading.Thread(target=_say, daemon=True).start()


# ── Shared UI helper ──────────────────────────────────────────────────────────
def _cancel_btn(parent, cmd):
    """Small red ✕ Cancel button placed at top-left."""
    btn = tk.Button(
        parent, text="✕  Cancel",
        font=F_SM, bg=DANGER, fg=TEXT,
        activebackground=DANGER, activeforeground=TEXT,
        relief="raised", bd=3, command=cmd,
    )
    btn.place(x=12, y=12)
    return btn


# ══════════════════════════════════════════════════════════════════════════════
#  SCREEN 1 — MAIN SCREEN
# ══════════════════════════════════════════════════════════════════════════════

class MainScreen(tk.Frame):
    def __init__(self, app):
        super().__init__(app, bg=BG)
        self._app = app

        tk.Label(self, text="PillWheel",
                 font=F_H1, fg=TEXT, bg=BG).pack(pady=(28, 4))
        tk.Label(self, text="Select a patient to begin collection",
                 font=F_BODY, fg=TEXT, bg=BG).pack(pady=(0, 16))

        grid = tk.Frame(self, bg=BG)
        grid.pack()

        for i, patient in enumerate(PATIENTS):
            row, col = divmod(i, 3)
            tk.Button(
                grid, text=patient["name"],
                font=F_BTN, bg=PRIMARY, fg=TEXT,
                activebackground=PRIMARY, activeforeground=TEXT,
                width=18, height=3, relief="raised", bd=4,
                command=lambda p=patient: self._select(p),
            ).grid(row=row, column=col, padx=8, pady=6)

        # Maintenance — small, bottom-right corner
        tk.Button(
            self, text="Maintenance",
            font=F_SM, bg=DISABLED, fg=TEXT,
            activebackground=DISABLED, activeforeground=TEXT,
            width=14, height=2, relief="raised", bd=2,
            command=lambda: app.show("MaintenanceScreen"),
        ).place(relx=1.0, rely=1.0, anchor="se", x=-12, y=-12)

    def _select(self, patient: dict):
        if not patient["encoding"]:
            self._unregistered_popup(patient["name"])
            return
        self._app.current_patient = patient
        self._app.show("CallingScreen")

    def _unregistered_popup(self, name: str):
        overlay = tk.Frame(self, bg="#2a2a45", relief="solid", bd=2)
        overlay.place(relx=0.5, rely=0.5, anchor="center", width=480, height=220)

        tk.Label(overlay, text="Patient Not Registered",
                 font=F_H2, fg=TEXT, bg="#2a2a45").pack(pady=(22, 8))
        tk.Label(overlay,
                 text=f"{name} has no registered face.\n"
                      "Please register in Maintenance Mode.",
                 font=F_BODY, fg=TEXT, bg="#2a2a45",
                 justify="center").pack()
        tk.Button(
            overlay, text="OK",
            font=F_BTN, bg=PRIMARY, fg=TEXT,
            activebackground=PRIMARY, activeforeground=TEXT,
            width=10, height=2, relief="raised", bd=3,
            command=overlay.destroy,
        ).pack(pady=16)


# ══════════════════════════════════════════════════════════════════════════════
#  SCREEN 2 — MAINTENANCE
# ══════════════════════════════════════════════════════════════════════════════

class MaintenanceScreen(tk.Frame):
    def __init__(self, app):
        super().__init__(app, bg=BG)
        self._app = app

        tk.Label(self, text="Maintenance",
                 font=F_H2, fg=TEXT, bg=BG).pack(pady=(20, 14))

        grid = tk.Frame(self, bg=BG)
        grid.pack()

        self._spk_btn = self._btn(grid, "Test Speaker",    self._test_speaker, 0, 0)
        self._s1_btn  = self._btn(grid, "Test Servo 1",    self._test_servo1,  0, 1)
        self._s2_btn  = self._btn(grid, "Test Servo 2",    self._test_servo2,  0, 2)
        self._btn(grid, "Register New Face", self._register_info,          1, 0, bg=DISABLED)
        self._btn(grid, "← Back to Main",   lambda: app.show("MainScreen"), 1, 2, bg=DANGER)

        self._status_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self._status_var,
                 font=F_BODY, fg=TEXT, bg=BG).pack(pady=14)

    @staticmethod
    def _btn(parent, text, cmd, row, col, bg=PRIMARY):
        b = tk.Button(
            parent, text=text, font=F_MAINT,
            bg=bg, fg=TEXT, activebackground=bg, activeforeground=TEXT,
            width=18, height=3, relief="raised", bd=4, command=cmd,
        )
        b.grid(row=row, column=col, padx=10, pady=8)
        return b

    def _set_status(self, text: str):
        self._status_var.set(text)

    def _test_speaker(self):
        self._spk_btn.config(state="disabled", bg=DISABLED)
        self._set_status("Playing...")

        def _work():
            phrase = "Testing speaker. One, two, three."
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 120)
                engine.say(phrase)
                engine.runAndWait()
                engine.stop()
            except Exception:
                try:
                    subprocess.call(["espeak", phrase])
                except Exception:
                    pass

            def _done():
                self._set_status("Speaker OK ✓")
                self._spk_btn.config(state="normal", bg=PRIMARY)
            self._app.after(0, _done)

        threading.Thread(target=_work, daemon=True).start()

    def _run_servo_test(self, btn, label, index):
        btn.config(state="disabled", bg=DISABLED)
        self._set_status(f"Rotating {label}...")

        def _work():
            self._app.servo.rotate_dispenser(index)

            def _done():
                self._set_status(f"{label} OK ✓")
                btn.config(state="normal", bg=PRIMARY)
            self._app.after(0, _done)

        threading.Thread(target=_work, daemon=True).start()

    def _test_servo1(self): self._run_servo_test(self._s1_btn, "Servo 1", 0)
    def _test_servo2(self): self._run_servo_test(self._s2_btn, "Servo 2", 1)

    def _register_info(self):
        self._set_status(
            "Use the Pi camera directly to register new patients. Coming soon.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — CALLING SCREEN
# ══════════════════════════════════════════════════════════════════════════════

class CallingScreen(tk.Frame):
    def __init__(self, app):
        super().__init__(app, bg=BG)
        self._app = app

        _cancel_btn(self, lambda: app.show("MainScreen"))

        self._msg_var = tk.StringVar()
        tk.Label(self, textvariable=self._msg_var,
                 font=F_H2, fg=TEXT, bg=BG,
                 wraplength=680, justify="center").pack(expand=True)

        tk.Button(
            self, text="Ready to Collect",
            font=F_BTN, bg=PRIMARY, fg=TEXT,
            activebackground=PRIMARY, activeforeground=TEXT,
            width=20, height=3, relief="raised", bd=4,
            command=lambda: app.show("FaceVerifyScreen"),
        ).pack(pady=(0, 60))

    def on_show(self):
        name = self._app.current_patient["name"]
        self._msg_var.set(f"{name}, your medication\nis ready for collection.")
        speak(f"{name}, please collect your medication.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — FACE VERIFY SCREEN
# ══════════════════════════════════════════════════════════════════════════════

class FaceVerifyScreen(tk.Frame):
    def __init__(self, app):
        super().__init__(app, bg=BG)
        self._app       = app
        self._anim_id   = None
        self._anim_step = 0

        self._cancel_btn = _cancel_btn(self, self._cancel)

        tk.Label(self, text="Please look at the camera.",
                 font=F_H2, fg=TEXT, bg=BG).pack(pady=(60, 20))

        self._scan_var = tk.StringVar()
        self._scan_lbl = tk.Label(self, textvariable=self._scan_var,
                                  font=F_BODY, fg=TEXT, bg=BG)
        self._scan_lbl.pack()

        # Shown only on verification failure
        self._return_btn = tk.Button(
            self, text="Return to Main",
            font=F_BTN, bg=DISABLED, fg=TEXT,
            activebackground=DISABLED, activeforeground=TEXT,
            width=18, height=3, relief="raised", bd=4,
            command=lambda: app.show("MainScreen"),
        )

    def on_show(self):
        # Reset to neutral state
        self.configure(bg=BG)
        self._scan_lbl.configure(bg=BG, fg=TEXT)
        self._return_btn.pack_forget()
        self._cancel_btn.place(x=12, y=12)
        self._start_animation()
        speak("Performing facial recognition. Please look at the camera.")
        threading.Thread(target=self._run_recognition, daemon=True).start()

    # ── Animation ─────────────────────────────────────────────────────────────

    def _start_animation(self):
        self._anim_step = 0
        self._animate()

    def _animate(self):
        dots = "." * ((self._anim_step % 3) + 1)
        self._scan_var.set(f"Scanning {dots}")
        self._anim_step += 1
        self._anim_id = self.after(500, self._animate)

    def _stop_animation(self):
        if self._anim_id:
            self.after_cancel(self._anim_id)
            self._anim_id = None

    # ── Recognition (background thread) ──────────────────────────────────────

    def _run_recognition(self):
        if not _FR_AVAILABLE:
            self._app.after(0, lambda: self._on_result(False))
            return

        patient = self._app.current_patient
        ref     = np.load(patient["encoding"])
        cap     = cv2.VideoCapture(CAMERA_INDEX)
        result  = False

        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                continue
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = _fr_lib.face_encodings(rgb)
            if encodings:
                if _fr_lib.compare_faces([ref], encodings[0], tolerance=0.5)[0]:
                    result = True
                    break

        cap.release()
        self._app.after(0, lambda: self._on_result(result))

    def _on_result(self, verified: bool):
        self._stop_animation()

        if verified:
            self._scan_var.set("")
            self.configure(bg=SUCCESS)
            self._scan_lbl.configure(bg=SUCCESS)
            speak("Identity verified.")
            self.after(1000, lambda: self._app.show("DispensingScreen"))
        else:
            self.configure(bg=DANGER)
            self._scan_lbl.configure(bg=DANGER)
            self._scan_var.set(
                "Verification failed. Please call for assistance.")
            speak("Verification failed. Please call for assistance.")
            self._cancel_btn.place_forget()
            self._return_btn.pack(pady=20)

    def _cancel(self):
        self._stop_animation()
        self._app.show("MainScreen")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — DISPENSING SCREEN
# ══════════════════════════════════════════════════════════════════════════════

class DispensingScreen(tk.Frame):
    def __init__(self, app):
        super().__init__(app, bg=BG)
        self._app = app

        tk.Label(self, text="Dispensing your medication.",
                 font=F_H2, fg=TEXT, bg=BG).pack(pady=(80, 8))
        tk.Label(self, text="Please wait.",
                 font=F_BODY, fg=TEXT, bg=BG).pack()

        self._progress_var = tk.StringVar()
        tk.Label(self, textvariable=self._progress_var,
                 font=F_BODY, fg=TEXT, bg=BG).pack(pady=16)

    def on_show(self):
        self._progress_var.set("")
        self._app._dispensing = True
        speak("Dispensing your medication.")
        threading.Thread(target=self._dispense, daemon=True).start()

    def _dispense(self):
        steps = [
            ("Dispensing 1 of 2...", 0),
            ("Dispensing 2 of 2...", 1),
        ]
        for label, index in steps:
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

        _cancel_btn(self, lambda: app.show("MainScreen"))

        tk.Label(self,
                 text="Please collect your medication\nfrom the tray.",
                 font=F_H2, fg=TEXT, bg=BG,
                 justify="center").pack(pady=(28, 12))

        reminders = tk.Frame(self, bg=BG)
        reminders.pack(pady=4)
        for line in (
            "💊  Take with food or milk",
            "💧  Take with a full glass of water",
            "⏰  Take at the same time each day",
        ):
            tk.Label(reminders, text=line,
                     font=("Arial", 22), fg=TEXT, bg=BG,
                     anchor="w").pack(fill="x", pady=2)

        btns = tk.Frame(self, bg=BG)
        btns.pack(pady=12)

        tk.Button(
            btns,
            text="✓  I have received the correct medication",
            font=F_BTN, bg=SUCCESS, fg=TEXT,
            activebackground=SUCCESS, activeforeground=TEXT,
            width=30, height=3, relief="raised", bd=4,
            command=lambda: app.show("SuccessScreen"),
        ).pack(pady=5)

        tk.Button(
            btns,
            text="✗  Something looks wrong",
            font=F_BTN, bg=DANGER, fg=TEXT,
            activebackground=DANGER, activeforeground=TEXT,
            width=30, height=3, relief="raised", bd=4,
            command=lambda: app.show("AssistanceScreen"),
        ).pack(pady=5)

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
                 font=F_H1, fg=TEXT, bg=SUCCESS,
                 justify="center").pack(expand=True)

    def on_show(self):
        patient = self._app.current_patient
        name    = patient["name"] if patient else "Patient"
        self._msg_var.set(f"Thank you, {name}.\nTake care!")
        speak("Thank you. Please take your medication as directed.")
        self.after(3000, lambda: self._app.show("MainScreen"))


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5b — ASSISTANCE SCREEN
# ══════════════════════════════════════════════════════════════════════════════

class AssistanceScreen(tk.Frame):
    def __init__(self, app):
        super().__init__(app, bg=DANGER)
        self._app = app

        tk.Label(self,
                 text="A carer has been notified.\nPlease wait.",
                 font=F_H1, fg=TEXT, bg=DANGER,
                 justify="center").pack(expand=True)

        tk.Button(
            self, text="Return to Main",
            font=F_BTN, bg="#2c2c2c", fg=TEXT,
            activebackground="#2c2c2c", activeforeground=TEXT,
            width=18, height=3, relief="raised", bd=4,
            command=lambda: app.show("MainScreen"),
        ).pack(pady=(0, 40))

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
#  APP ROOT
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    _SCREENS = (
        MainScreen,
        MaintenanceScreen,
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
