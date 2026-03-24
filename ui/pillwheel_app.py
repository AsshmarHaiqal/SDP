"""
ui/pillwheel_app.py — PillWheelApp: the thin orchestrator.

Responsibilities:
  - Build the UI (delegates to ui/screens.py)
  - Own all hardware instances
  - Wire FaceFlow and DispenseFlow to UI callbacks
  - Manage the live camera feed display

Consumes the UNIFIED patient dict from api_client.py — reads both
winterscone keys (firstName, medicineName, quantity) and asshmarhaiqal
keys (display_name, medicine_name, pill_count) via safe fallbacks.

Hardware:
  electronic/servo_controller.py  — ServoController
  electronic/pill_recogniser.py   — PillRecogniser
  electronic/tray_sweep.py        — sweep()
  electronic/sound_actuator.py    — SoundActuator
  face_tracking (optional)        — servo + Haar face scan
  api_client.py                   — APIClient
"""

import os
import threading
from datetime import datetime

import tkinter as tk
from tkinter import font as tkfont

try:
    from PIL import Image, ImageTk
    _PIL = True
except ImportError:
    _PIL = False

from .theme import (
    C_BG, C_PANEL, C_MUTED,
    W, H,
    F_TITLE, F_BODY, F_BTN, F_STATUS, F_SMALL,
)
from .camera_manager import CameraManager
from .face_flow      import FaceFlow
from .dispense_flow  import DispenseFlow
from . import screen as scr


class PillWheelApp:
    """
    Touchscreen orchestrator for PillWheel.

    Parameters
    ----------
    root        : tk.Tk
    servo       : ServoController
    pill_rec    : PillRecogniser
    tray_sweep  : callable
    sound       : SoundActuator
    api_client  : APIClient
    ft          : face_tracking module or None
    fr          : FacialRecognition instance or None  (local mode only)
    fr_mode     : "server" | "local"
    audit_dir   : path for audit images
    on_maintenance : callable or None — opens the maintenance UI
    """

    COMPLETE_DELAY = 5
    ERROR_DELAY    = 10

    def __init__(self, root, servo, pill_rec, tray_sweep, sound, api_client,
                 ft=None, fr=None, fr_mode: str = "server",
                 audit_dir="data/audit", on_maintenance=None):
        self.root    = root
        self._sound  = sound
        self._api    = api_client
        self._on_maintenance = on_maintenance

        # ── Camera ────────────────────────────────────────────────────────────
        self._cam = CameraManager(face_tracking_module=ft)

        # ── Flow controllers ──────────────────────────────────────────────────
        self._face_flow = FaceFlow(
            camera=self._cam,
            api_client=api_client,
            servo=servo,
            root=root,
            ft=ft,
            fr=fr,
            fr_mode=fr_mode,
        )
        self._disp_flow = DispenseFlow(
            servo=servo,
            pill_rec=pill_rec,
            tray_sweep=tray_sweep,
            sound=sound,
            camera=self._cam,
            api_client=api_client,
            root=root,
            audit_dir=audit_dir,
        )

        # ── Session state ─────────────────────────────────────────────────────
        self.current_patient = None
        self._feed_label     = None

        # ── Build UI ──────────────────────────────────────────────────────────
        self._fonts   = self._make_fonts()
        self._screens = {}
        self._refs    = {}   # StringVars / widget refs from screen builders
        self._build_ui()
        self._set_status("System ready")
        self.show_screen("home")

    # ── Font helpers ──────────────────────────────────────────────────────────

    def _make_fonts(self) -> dict:
        return {
            "title":  tkfont.Font(family="Helvetica", size=F_TITLE, weight="bold"),
            "body":   tkfont.Font(family="Helvetica", size=F_BODY),
            "btn":    tkfont.Font(family="Helvetica", size=F_BTN,   weight="bold"),
            "status": tkfont.Font(family="Helvetica", size=F_STATUS),
            "small":  tkfont.Font(family="Helvetica", size=F_SMALL),
        }

    # ── Status bar ────────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._status_var.set(f"{ts}   {msg}")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._status_var = tk.StringVar()

        outer = tk.Frame(self.root, bg=C_BG)
        outer.pack(fill="both", expand=True)

        tk.Label(
            self.root, textvariable=self._status_var,
            bg=C_PANEL, fg=C_MUTED,
            font=self._fonts["status"],
            anchor="w", padx=12, pady=4,
        ).pack(side="bottom", fill="x")

        container = tk.Frame(outer, bg=C_BG)
        container.pack(fill="both", expand=True)

        names = ("home", "scanning", "verified", "dispensing", "complete", "error")
        for name in names:
            f = tk.Frame(container, bg=C_BG)
            f.place(relx=0, rely=0, relwidth=1, relheight=1)
            self._screens[name] = f

        # Build each screen and capture widget refs
        scr.build_home(
            self._screens["home"], self._fonts,
            self._on_start_pressed,
            on_maintenance=self._on_maintenance,
        )

        self._refs["scan"] = scr.build_scanning(
            self._screens["scanning"], self._fonts, self._cancel_to_home
        )
        self._refs["verified"] = scr.build_verified(
            self._screens["verified"], self._fonts
        )
        self._refs["disp"] = scr.build_dispensing(
            self._screens["dispensing"], self._fonts
        )
        self._refs["complete"] = scr.build_complete(
            self._screens["complete"], self._fonts
        )
        self._refs["error"] = scr.build_error(
            self._screens["error"], self._fonts
        )

    def show_screen(self, name: str) -> None:
        self._screens[name].tkraise()

    # ── Live camera feed ──────────────────────────────────────────────────────

    def _start_feed(self, label: tk.Label) -> None:
        self._feed_label = label
        self._update_feed()

    def _stop_feed(self) -> None:
        self._feed_label = None

    def _update_feed(self) -> None:
        if self._feed_label is None:
            return
        if _PIL and self._cam.latest_frame is not None:
            try:
                frame   = self._cam.latest_frame
                h, w    = frame.shape[:2]
                scale   = min(640 / w, 300 / h)
                nw, nh  = int(w * scale), int(h * scale)
                import cv2
                resized = cv2.resize(frame, (nw, nh))
                rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                photo   = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                self._feed_label.config(image=photo)
                self._feed_label.image = photo
            except Exception:
                pass
        self.root.after(100, self._update_feed)

    # ── Heading dots animation ────────────────────────────────────────────────

    def _start_dots(self) -> None:
        self._dot_n = 0
        self._tick_dots()

    def _tick_dots(self) -> None:
        dots = "." * ((self._dot_n % 3) + 1)
        self._refs["scan"]["heading"].set(f"Scanning for face{dots}")
        self._dot_n += 1
        self._dot_job = self.root.after(500, self._tick_dots)

    def _stop_dots(self) -> None:
        if hasattr(self, "_dot_job"):
            self.root.after_cancel(self._dot_job)

    # ── Countdown helper ──────────────────────────────────────────────────────

    def _countdown_to_home(self, seconds: int, cd_var: tk.StringVar) -> None:
        if seconds <= 0:
            self._go_home()
            return
        cd_var.set(f"Returning to home in {seconds}s…")
        self.root.after(1000, lambda: self._countdown_to_home(seconds - 1, cd_var))

    def _go_home(self) -> None:
        self._stop_feed()
        self.current_patient = None
        self.show_screen("home")
        self._set_status("System ready")

    def _cancel_to_home(self) -> None:
        self._face_flow.stop()
        self._disp_flow.stop()
        self._stop_dots()
        self._go_home()

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 1: Start scan
    # ══════════════════════════════════════════════════════════════════════════

    def _on_start_pressed(self) -> None:
        self._cam.open()
        self._cam.start_loop()
        self._start_feed(self._refs["scan"]["feed"])
        self.show_screen("scanning")
        self._start_dots()
        self._set_status("Scanning for face…")
        self._sound.verifying_face()
        self._face_flow.start(
            on_identified=self._on_identified,
            on_failed=self._on_face_failed,
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 2: Face scan callbacks (main thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _on_face_failed(self, reason: str) -> None:
        self._stop_dots()
        self._sound.access_denied()
        self._show_error(reason)

    def _on_identified(self, patient: dict) -> None:
        """
        Called with the UNIFIED patient dict.
        Reads both winterscone and asshmarhaiqal keys via safe fallbacks.
        """
        self._stop_dots()
        self._stop_feed()
        self.current_patient = patient

        rx = patient["prescriptions"][0]

        # Names — try winterscone keys first, fall back to asshmarhaiqal
        first_name = (
            patient.get("firstName")
            or patient.get("patientName", "there").split()[0]
        )
        full_name = (
            f"{patient.get('firstName', '')} {patient.get('lastName', '')}".strip()
            or patient.get("display_name", f"Patient {patient.get('patientId', '')}")
        )

        # Prescription info
        med_name  = rx.get("medicineName") or rx.get("medicine_name", "Unknown")
        quantity  = rx.get("quantity") or rx.get("pill_count", 1)

        self._sound.verified()
        self._sound.speak(f"Welcome, {first_name}")

        self._refs["verified"]["name"].set(f"Welcome, {full_name}")
        self._refs["verified"]["rx"].set(f"{med_name}  ×  {quantity}")
        self.show_screen("verified")
        self._set_status(f"Identified: {full_name}")
        self.root.after(2000, self._start_dispensing)

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 3: Dispensing
    # ══════════════════════════════════════════════════════════════════════════

    def _start_dispensing(self) -> None:
        if self.current_patient is None:
            return
        self._start_feed(self._refs["disp"]["feed"])
        self.show_screen("dispensing")
        self._sound.dispensing()

        self._disp_flow.start(
            patient=self.current_patient,
            on_status=self._on_dispense_status,
            on_complete=self._on_dispense_complete,
            on_error=lambda msg: self._show_error(msg),
        )

    def _on_dispense_status(self, msg: str) -> None:
        self._refs["disp"]["status"].set(msg)
        self._set_status(msg)

    def _on_dispense_complete(self, patient: dict, rx: dict, timestamp: str) -> None:
        self._stop_feed()

        full_name = (
            f"{patient.get('firstName', '')} {patient.get('lastName', '')}".strip()
            or patient.get("display_name", "Patient")
        )
        med_name = rx.get("medicineName") or rx.get("medicine_name", "")
        quantity = rx.get("quantity") or rx.get("pill_count", 1)

        ts_display = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime(
            "%H:%M  %d/%m/%Y"
        )

        self._refs["complete"]["name"].set(full_name)
        self._refs["complete"]["details"].set(
            f"{med_name}  ×  {quantity}  —  {ts_display}"
        )
        self._refs["complete"]["cd"].set("")
        self.show_screen("complete")
        self._set_status(f"Dispense complete for {full_name}")
        self._countdown_to_home(self.COMPLETE_DELAY, self._refs["complete"]["cd"])

    # ══════════════════════════════════════════════════════════════════════════
    #  Error screen
    # ══════════════════════════════════════════════════════════════════════════

    def _show_error(self, message: str) -> None:
        self._stop_feed()
        self._refs["error"]["msg"].set(message)
        self._refs["error"]["cd"].set("")
        self.show_screen("error")
        self._set_status(f"Error: {message.splitlines()[0]}")
        self._countdown_to_home(self.ERROR_DELAY, self._refs["error"]["cd"])