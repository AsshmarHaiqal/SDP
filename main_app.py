"""
SDP/main_app.py
PillWheel — Raspberry Pi Touchscreen UI  (800 × 480)

Screen map
----------
  Home              Maintenance Mode | Simulation Mode (Phase 2)
  Maintenance       Servo Tests | Camera Tests | Buzzer Test
  ServoTest         Dispenser D0–D13 grid, Camera servo, Tray servo
  CameraTest        Facial Recognition branch | Pill Detection branch
  FacialTest        Register new user | Mock live verification
  PillDetectionTest Live preview (cv2 window) | Single count test | param2 tuner
  Buzzer            Placeholder

Deployment
----------
  Dev / laptop : CAMERA_INDEX = 0  |  FULLSCREEN = False
  Raspberry Pi : CAMERA_INDEX = 1  |  FULLSCREEN = True

Run from SDP/:
    python main_app.py
"""

import threading
import tkinter as tk

# ── Deployment settings ───────────────────────────────────────────────────────
CAMERA_INDEX = 0       # 0 = laptop built-in  |  1 = Pi USB webcam
FULLSCREEN   = False   # True on Raspberry Pi

# Patch camera indices before any module-level camera reference executes.
import electronic.facial_recognition as _fr_mod
import electronic.pill_detection      as _pd_mod
_fr_mod.CAMERA_INDEX   = CAMERA_INDEX
_pd_mod._CAMERA_INDEX  = CAMERA_INDEX

import cv2
import numpy as np

from config.hardware_config import (
    DISPENSER_COUNT,
    SPECIAL_SERVO_CHANNELS,
    CAMERA_TILT_ANGLE,
    CAMERA_RETURN_ANGLE,
    TRAY_TILT_ANGLE,
)
from electronic.servo_controller    import ServoController
from electronic.pill_detection      import PillDetector
from electronic.facial_recognition  import enroll_face, verify_access_live, list_enrolled

# ── Theme ─────────────────────────────────────────────────────────────────────
BG      = "#1c1c2e"
HDR_BG  = "#16213e"
SEP_CLR = "#2a2a45"
BLU     = "#0f3460"
GRN     = "#1b5e20"
RED     = "#b71c1c"
AMB     = "#bf360c"
GRY     = "#37474f"
TEL     = "#006064"
FG      = "#eceff1"
FG_DIM  = "#78909c"
OK_FG   = "#69f0ae"
ERR_FG  = "#ff5252"

# Fonts  (Helvetica renders cleanly on both Pi and macOS)
TF  = ("Helvetica", 20, "bold")   # page title
BF  = ("Helvetica", 16, "bold")   # large buttons
BSF = ("Helvetica", 13, "bold")   # small buttons / section headers
LF  = ("Helvetica", 13)           # body labels
SF  = ("Helvetica", 12, "italic") # dim / status text

W, H = 800, 480


# ── Shared UI helpers ─────────────────────────────────────────────────────────

def _btn(parent, text, cmd, bg=BLU, fg=FG, font=None, **kw):
    """Flat, touch-friendly button."""
    return tk.Button(
        parent, text=text, command=cmd,
        bg=bg, fg=fg, activebackground=bg, activeforeground=fg,
        font=font or BF, relief="flat", bd=0,
        highlightthickness=0, cursor="hand2", **kw,
    )


def _hdr(parent, title, back_cmd=None):
    """56-px header bar with optional Back button on the right."""
    bar = tk.Frame(parent, bg=HDR_BG, height=56)
    bar.pack(fill="x")
    bar.pack_propagate(False)
    tk.Label(bar, text=title, bg=HDR_BG, fg=FG,
             font=TF, anchor="w").pack(side="left", padx=16, pady=8)
    if back_cmd:
        _btn(bar, "← Back", back_cmd, bg=GRY, font=BSF,
             padx=12, pady=4).pack(side="right", padx=12, pady=8)
    return bar


def _sep(parent):
    tk.Frame(parent, bg=SEP_CLR, height=2).pack(fill="x", padx=12, pady=6)


def _sec(parent, text):
    """Dim section-heading label."""
    tk.Label(parent, text=text, bg=BG, fg=FG_DIM,
             font=SF, anchor="w").pack(fill="x", padx=16, pady=(6, 2))


# ── Page base ─────────────────────────────────────────────────────────────────

class Page(tk.Frame):
    """All screens inherit from this.  on_show() is called before tkraise()."""

    def on_show(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  HOME
# ═══════════════════════════════════════════════════════════════════════════════

class HomePage(Page):
    def __init__(self, app: "PillWheelApp"):
        super().__init__(app, bg=BG)

        _hdr(self, "💊  PillWheel")

        body = tk.Frame(self, bg=BG)
        body.pack(expand=True, padx=100)

        _btn(body, "🔧   Maintenance Mode",
             lambda: app.show("ServoTestPage" if False else "MaintenancePage"),
             bg=AMB, height=3).pack(fill="x", pady=16)

        _btn(body, "▶   Simulation Mode",
             lambda: None, bg=GRY, height=3,
             state="disabled").pack(fill="x", pady=16)

        tk.Label(body, text="Simulation mode — Phase 2, coming soon.",
                 bg=BG, fg=FG_DIM, font=SF).pack()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAINTENANCE MENU
# ═══════════════════════════════════════════════════════════════════════════════

class MaintenancePage(Page):
    def __init__(self, app: "PillWheelApp"):
        super().__init__(app, bg=BG)

        _hdr(self, "🔧  Maintenance Mode",
             back_cmd=lambda: app.show("HomePage"))

        body = tk.Frame(self, bg=BG)
        body.pack(expand=True, padx=100)

        _btn(body, "⚙️   Servo Tests",
             lambda: app.show("ServoTestPage"),
             bg=BLU, height=3).pack(fill="x", pady=12)

        _btn(body, "📷   Camera Tests",
             lambda: app.show("CameraTestPage"),
             bg=TEL, height=3).pack(fill="x", pady=12)

        _btn(body, "🔔   Buzzer Test",
             lambda: app.show("BuzzerPage"),
             bg=GRY, height=3, state="disabled").pack(fill="x", pady=12)

        tk.Label(body, text="Buzzer — functionality coming soon.",
                 bg=BG, fg=FG_DIM, font=SF).pack()


# ═══════════════════════════════════════════════════════════════════════════════
#  SERVO TEST
# ═══════════════════════════════════════════════════════════════════════════════

class ServoTestPage(Page):
    def __init__(self, app: "PillWheelApp"):
        super().__init__(app, bg=BG)
        self._app  = app
        self._busy = False

        _hdr(self, "⚙️  Servo Tests",
             back_cmd=lambda: app.show("MaintenancePage"))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=2)

        # ── Dispenser grid ───���────────────────────────────────────────────────
        _sec(body, f"DISPENSER SLOTS  (0 – {DISPENSER_COUNT - 1})  "
                   "— press to run one full 0° → 180° → 0° sweep")

        grid = tk.Frame(body, bg=BG)
        grid.pack(fill="x", padx=8, pady=(2, 6))

        for i in range(DISPENSER_COUNT):
            row, col = divmod(i, 7)
            # default-arg capture avoids late-binding
            tk.Button(
                grid, text=f"D{i}",
                command=lambda idx=i: self._fire(
                    lambda: app.servo.rotate_dispenser(idx)),
                bg=BLU, fg=FG, activebackground="#1a4f8a", activeforeground=FG,
                font=BSF, relief="flat", bd=0,
                width=7, height=2, highlightthickness=0, cursor="hand2",
            ).grid(row=row, column=col, padx=3, pady=3, sticky="nsew")

        _sep(body)

        # ── Camera servo ──────────────────────────────────────────────────────
        _sec(body, f"CAMERA SERVO  (PCA9685 ch {SPECIAL_SERVO_CHANNELS[0]})")

        cam_row = tk.Frame(body, bg=BG)
        cam_row.pack(fill="x", padx=8, pady=(2, 6))

        _btn(cam_row, f"▼  Tilt Down  ({CAMERA_TILT_ANGLE}°)",
             lambda: self._fire(lambda: app.servo.set_servo_angle(
                 SPECIAL_SERVO_CHANNELS[0], CAMERA_TILT_ANGLE)),
             bg=TEL, font=BSF, width=26, height=2,
             ).pack(side="left", padx=(0, 8))

        _btn(cam_row, f"▲  Return Forward  ({CAMERA_RETURN_ANGLE}°)",
             lambda: self._fire(lambda: app.servo.set_servo_angle(
                 SPECIAL_SERVO_CHANNELS[0], CAMERA_RETURN_ANGLE)),
             bg=GRN, font=BSF, width=26, height=2,
             ).pack(side="left")

        _sep(body)

        # ── Tray servo ────────────────────────────────────────────────────────
        _sec(body, f"TRAY SERVO  (PCA9685 ch {SPECIAL_SERVO_CHANNELS[1]})")

        tray_row = tk.Frame(body, bg=BG)
        tray_row.pack(fill="x", padx=8, pady=(2, 6))

        _btn(tray_row, f"↗  Tilt  ({TRAY_TILT_ANGLE}°)",
             lambda: self._fire(lambda: app.servo.set_servo_angle(
                 SPECIAL_SERVO_CHANNELS[1], TRAY_TILT_ANGLE)),
             bg=AMB, font=BSF, width=26, height=2,
             ).pack(side="left", padx=(0, 8))

        _btn(tray_row, "↘  Return Flat  (0°)",
             lambda: self._fire(lambda: app.servo.set_servo_angle(
                 SPECIAL_SERVO_CHANNELS[1], 0)),
             bg=GRN, font=BSF, width=26, height=2,
             ).pack(side="left")

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Select a servo to test.")
        self._status_lbl = tk.Label(
            body, textvariable=self._status_var,
            bg=BG, fg=FG_DIM, font=SF, anchor="w")
        self._status_lbl.pack(fill="x", padx=12, pady=4)

    def _fire(self, fn):
        """Run *fn* in a daemon thread; block re-entry while busy."""
        if self._busy:
            return
        self._busy = True
        self._status_var.set("Rotating…")
        self._status_lbl.config(fg=FG_DIM)

        def _work():
            fn()
            self._app.after(0, self._done)

        threading.Thread(target=_work, daemon=True).start()

    def _done(self):
        self._busy = False
        self._status_var.set("Done ✓")
        self._status_lbl.config(fg=OK_FG)


# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA TEST MENU
# ═══════════════════════════════════════════════════════════════════════════════

class CameraTestPage(Page):
    def __init__(self, app: "PillWheelApp"):
        super().__init__(app, bg=BG)

        _hdr(self, "📷  Camera Tests",
             back_cmd=lambda: app.show("MaintenancePage"))

        body = tk.Frame(self, bg=BG)
        body.pack(expand=True, padx=100)

        _btn(body, "👤   Facial Recognition Tests",
             lambda: app.show("FacialTestPage"),
             bg=BLU, height=3).pack(fill="x", pady=14)

        _btn(body, "💊   Pill Detection Tests",
             lambda: app.show("PillDetectionTestPage"),
             bg=TEL, height=3).pack(fill="x", pady=14)


# ═══════════════════════════════════════════════════════════════════════════════
#  FACIAL RECOGNITION TEST
# ═══════════════════════════════════════════════════════════════════════════════

class FacialTestPage(Page):
    def __init__(self, app: "PillWheelApp"):
        super().__init__(app, bg=BG)
        self._app = app

        _hdr(self, "👤  Facial Recognition",
             back_cmd=lambda: app.show("CameraTestPage"))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=8)

        # ── Register ──────────────────────────────────────────────────────────
        _sec(body, "REGISTER NEW USER  — point camera at face, then press Capture")

        reg_row = tk.Frame(body, bg=BG)
        reg_row.pack(fill="x", padx=8, pady=4)

        tk.Label(reg_row, text="Name:", bg=BG, fg=FG,
                 font=LF).pack(side="left", padx=(0, 8))

        self._name_entry = tk.Entry(
            reg_row, font=LF, width=18,
            bg="#263238", fg=FG, insertbackground=FG,
            relief="flat", bd=6)
        self._name_entry.pack(side="left", padx=(0, 14))

        _btn(reg_row, "📸  Capture & Register",
             self._register, bg=GRN, font=BSF,
             padx=12, pady=6).pack(side="left")

        _sep(body)

        # ── Verify ────────────────────────────────────────────────────────────
        _sec(body, "MOCK VERIFICATION  — opens live camera; green box = verified")

        ver_row = tk.Frame(body, bg=BG)
        ver_row.pack(fill="x", padx=8, pady=4)

        tk.Label(ver_row, text="User:", bg=BG, fg=FG,
                 font=LF).pack(side="left", padx=(0, 8))

        self._enrolled_var = tk.StringVar(value="")
        self._user_menu = tk.OptionMenu(ver_row, self._enrolled_var, "")
        self._user_menu.config(
            bg="#263238", fg=FG, font=LF, relief="flat",
            bd=0, highlightthickness=0, width=16,
            activebackground=BLU, activeforeground=FG)
        self._user_menu["menu"].config(
            bg="#263238", fg=FG, font=LF,
            activebackground=BLU, activeforeground=FG)
        self._user_menu.pack(side="left", padx=(0, 12))

        _btn(ver_row, "👁  Verify (Live)",
             self._verify, bg=AMB, font=BSF,
             padx=12, pady=6).pack(side="left", padx=(0, 8))

        _btn(ver_row, "↻  Refresh",
             self.on_show, bg=GRY, font=BSF,
             padx=8, pady=6).pack(side="left")

        _sep(body)

        self._status_var = tk.StringVar(value="")
        self._status_lbl = tk.Label(
            body, textvariable=self._status_var,
            bg=BG, fg=FG_DIM, font=LF, anchor="w")
        self._status_lbl.pack(fill="x", padx=8, pady=6)

    def on_show(self):
        """Refresh enrolled-user dropdown each time this page is raised."""
        enrolled = list_enrolled()
        menu = self._user_menu["menu"]
        menu.delete(0, "end")
        if enrolled:
            for name in enrolled:
                menu.add_command(
                    label=name,
                    command=lambda v=name: self._enrolled_var.set(v))
            self._enrolled_var.set(enrolled[0])
            self._set_status(f"{len(enrolled)} enrolled user(s) found.")
        else:
            menu.add_command(label="(none)", command=lambda: None)
            self._enrolled_var.set("")
            self._set_status("No users enrolled yet. Register one above.")

    def _set_status(self, text, ok=None):
        self._status_var.set(text)
        color = (OK_FG if ok else ERR_FG) if ok is not None else FG_DIM
        self._status_lbl.config(fg=color)

    def _register(self):
        name = self._name_entry.get().strip()
        if not name:
            self._set_status("⚠  Enter a name first.", ok=False)
            return
        self._set_status(f"Capturing face for '{name}' — look at camera…")

        def _work():
            result = enroll_face(name)
            if result:
                self._app.after(0, lambda: (
                    self._set_status(
                        f"✓  '{name}' registered successfully.", ok=True),
                    self.on_show(),
                ))
            else:
                self._app.after(0, lambda: self._set_status(
                    "✗  No face detected. Check lighting and try again.",
                    ok=False))

        threading.Thread(target=_work, daemon=True).start()

    def _verify(self):
        name = self._enrolled_var.get().strip()
        if not name:
            self._set_status(
                "⚠  No user selected. Register first or press Refresh.",
                ok=False)
            return
        self._set_status(
            f"Verifying '{name}' — look at camera.  "
            "Press q in camera window to cancel.")

        def _work():
            result = verify_access_live(name)
            msg = (f"✓  VERIFIED — '{name}' recognised."
                   if result else
                   f"✗  NOT RECOGNISED — '{name}'.")
            self._app.after(0, lambda: self._set_status(msg, ok=result))

        threading.Thread(target=_work, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
#  PILL DETECTION TEST
# ═══════════════════════════════════════════════════════════════════════════════

class PillDetectionTestPage(Page):
    def __init__(self, app: "PillWheelApp"):
        super().__init__(app, bg=BG)
        self._app             = app
        self._preview_running = False
        self._preview_stop    = threading.Event()

        _hdr(self, "💊  Pill Detection Tests",
             back_cmd=lambda: (self._stop_preview(),
                               app.show("CameraTestPage")))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=8)

        # ── Detection sensitivity tuner ───────────────────────────────────────
        _sec(body,
             "SENSITIVITY  (param2:  lower → more sensitive, higher → stricter)")

        tune_row = tk.Frame(body, bg=BG)
        tune_row.pack(fill="x", padx=8, pady=4)

        _btn(tune_row, "−", self._p2_dec,
             bg=GRY, font=BF, width=3, padx=2).pack(side="left")

        self._p2_var = tk.IntVar(value=_pd_mod._HOUGH_PARAM2)
        tk.Label(tune_row, textvariable=self._p2_var,
                 bg=BG, fg=FG, font=BF, width=4).pack(side="left", padx=6)

        _btn(tune_row, "+", self._p2_inc,
             bg=GRY, font=BF, width=3, padx=2).pack(side="left")

        tk.Label(tune_row, text="  default: 30   |   range: 4 – 100",
                 bg=BG, fg=FG_DIM, font=SF).pack(side="left", padx=14)

        _sep(body)

        # ── Test buttons ──────────────────────────────────────────────────────
        btn_row = tk.Frame(body, bg=BG)
        btn_row.pack(fill="x", padx=8, pady=8)

        self._preview_btn = _btn(
            btn_row, "▶  Live Preview",
            self._toggle_preview,
            bg=TEL, font=BSF, width=22, height=3)
        self._preview_btn.pack(side="left", padx=(0, 16))

        _btn(btn_row,
             "📸  Count Test\n(single capture + save)",
             self._count_test,
             bg=BLU, font=BSF, width=22, height=3).pack(side="left")

        _sep(body)

        # ── Status ────────────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready.")
        self._status_lbl = tk.Label(
            body, textvariable=self._status_var,
            bg=BG, fg=FG_DIM, font=LF, anchor="w")
        self._status_lbl.pack(fill="x", padx=8, pady=4)

    # ── param2 tuning ─────────────────────────────────────────────────────────

    def _p2_inc(self):
        v = min(self._p2_var.get() + 2, 100)
        self._p2_var.set(v)
        _pd_mod._HOUGH_PARAM2 = v

    def _p2_dec(self):
        v = max(self._p2_var.get() - 2, 4)
        self._p2_var.set(v)
        _pd_mod._HOUGH_PARAM2 = v

    # ── status helper ─────────────────────────────────────────────────────────

    def _set_status(self, text, ok=None):
        self._status_var.set(text)
        color = (OK_FG if ok else ERR_FG) if ok is not None else FG_DIM
        self._status_lbl.config(fg=color)

    # ── Live preview ──────────────────────────────────────────────────────────

    def _toggle_preview(self):
        if self._preview_running:
            self._stop_preview()
        else:
            self._preview_stop.clear()
            self._preview_running = True
            self._preview_btn.config(text="⏹  Stop Preview", bg=RED)
            self._set_status(
                "Preview running — press Stop or 'q' in the camera window.")
            threading.Thread(target=self._preview_worker, daemon=True).start()

    def _stop_preview(self):
        if self._preview_running:
            self._preview_stop.set()

    def _preview_worker(self):
        """
        Opens a cv2 window with Hough circles drawn live.
        Runs in a daemon thread (fine on Linux/Pi; on macOS use Count Test).
        """
        cap = cv2.VideoCapture(CAMERA_INDEX)
        while not self._preview_stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp        = _pd_mod._HOUGH_DP,
                minDist   = _pd_mod._HOUGH_MIN_DIST,
                param1    = _pd_mod._HOUGH_PARAM1,
                param2    = _pd_mod._HOUGH_PARAM2,
                minRadius = _pd_mod._HOUGH_MIN_RADIUS,
                maxRadius = _pd_mod._HOUGH_MAX_RADIUS,
            )
            count   = 0 if circles is None else len(circles[0])
            overlay = frame.copy()
            if circles is not None:
                for (x, y, r) in np.uint16(np.around(circles[0])):
                    cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)
                    cv2.circle(overlay, (x, y), 2,  (0, 0, 255), 3)

            cv2.putText(
                overlay,
                f"Detected: {count}   param2={_pd_mod._HOUGH_PARAM2}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Pill Detection — Preview  (q to stop)", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self._app.after(0, self._on_preview_stopped)

    def _on_preview_stopped(self):
        self._preview_running = False
        self._preview_btn.config(text="▶  Live Preview", bg=TEL)
        self._set_status("Preview stopped.")

    # ── Single count test ─────────────────────────────────────────────────────

    def _count_test(self):
        self._set_status("Capturing image…")

        def _work():
            img, path = self._app.detector._capture_tray_image(
                "maint_test", "count_test", 0, 0)
            count = self._app.detector._count_pills(img)
            self._app.after(0, lambda: self._set_status(
                f"✓  Detected {count} pill(s).    Saved: {path}", ok=True))

        threading.Thread(target=_work, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
#  BUZZER PLACEHOLDER
# ═══════════════════════════════════════════════════════════════════════════════

class BuzzerPage(Page):
    def __init__(self, app: "PillWheelApp"):
        super().__init__(app, bg=BG)

        _hdr(self, "🔔  Buzzer Test",
             back_cmd=lambda: app.show("MaintenancePage"))

        body = tk.Frame(self, bg=BG)
        body.pack(expand=True)

        tk.Label(body, text="🔔", bg=BG, fg=FG_DIM,
                 font=("Helvetica", 56)).pack(pady=20)

        tk.Label(body, text="Buzzer functionality — coming soon",
                 bg=BG, fg=FG, font=BF).pack()

        tk.Label(body,
                 text="GPIO pin assignment and trigger logic\n"
                      "will be implemented in a later phase.",
                 bg=BG, fg=FG_DIM, font=LF, justify="center").pack(pady=10)


# ═══════════════════════════════════════════════════════════════════════════════
#  APPLICATION ROOT
# ═══════════════════════════════════════════════════════════════════════════════

class PillWheelApp(tk.Tk):

    _PAGE_CLASSES = [
        # Listed in stacking order — last = on top before first show()
        BuzzerPage,
        PillDetectionTestPage,
        FacialTestPage,
        CameraTestPage,
        ServoTestPage,
        MaintenancePage,
        HomePage,
    ]

    def __init__(self):
        super().__init__()
        self.title("PillWheel")
        self.configure(bg=BG)
        self.geometry(f"{W}x{H}")
        self.resizable(False, False)

        if FULLSCREEN:
            self.attributes("-fullscreen", True)
            # Escape key exits full-screen (useful during maintenance)
            self.bind("<Escape>",
                      lambda e: self.attributes("-fullscreen", False))

        # Shared hardware (ServoController gracefully sims when no hardware)
        self.servo    = ServoController()
        self.detector = PillDetector(self.servo)

        # Build all pages and stack them with place()
        self._pages: dict[str, Page] = {}
        for Cls in self._PAGE_CLASSES:
            page = Cls(self)
            self._pages[Cls.__name__] = page
            page.place(x=0, y=0, relwidth=1, relheight=1)

        self.show("HomePage")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def show(self, name: str):
        """Raise page *name* and call its on_show() hook."""
        page = self._pages[name]
        page.on_show()
        page.tkraise()

    def _on_close(self):
        self.servo.cleanup()
        self.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PillWheelApp().mainloop()
