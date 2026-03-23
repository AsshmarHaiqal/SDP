"""
firmware/servo_8_test.py
8 buttons — each rotates the corresponding PCA9685 servo (ch 0-7): 0 → 180 → 0

Usage:
    export DISPLAY=:0
    python3 firmware/servo_8_test.py
"""

import sys
import os
import threading
import time

import tkinter as tk
from tkinter import font as tkfont

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from adafruit_servokit import ServoKit
    kit = ServoKit(channels=16)
    for ch in range(8):
        kit.servo[ch].set_pulse_width_range(500, 2500)
    HW = True
    print("ServoKit ready ✔  (channels 0–7 initialised)")
except Exception as e:
    HW = False
    print(f"No hardware ({e}) — simulation mode")


def rotate(channel: int, status_var: tk.StringVar, btn: tk.Button, busy: list):
    """Rotate servo 0→180→0 in a background thread."""
    def _run():
        busy[0] = True
        if HW:
            kit.servo[channel].angle = 0;   time.sleep(0.5)
            kit.servo[channel].angle = 180; time.sleep(0.5)
            kit.servo[channel].angle = 0;   time.sleep(0.5)
        else:
            print(f"[sim] ch{channel}: 0→180→0")
            time.sleep(1.5)
        busy[0] = False
        root.after(0, lambda: _done(channel, status_var, btn))

    threading.Thread(target=_run, daemon=True).start()


def _done(channel, status_var, btn):
    status_var.set(f"ch{channel} done — ready")
    btn.config(state="normal", bg=COLOURS[channel % len(COLOURS)])


COLOURS = [
    "#388bfd", "#9b59b6", "#238636", "#d97706",
    "#e74c3c", "#1abc9c", "#e67e22", "#2980b9",
]

root = tk.Tk()
root.title("Servo Test — ch 0–7")
root.geometry("800x480")
root.configure(bg="#0d1117")
root.resizable(True, True)

f_title  = tkfont.Font(family="Courier", size=14, weight="bold")
f_btn    = tkfont.Font(family="Courier", size=16, weight="bold")
f_status = tkfont.Font(family="Courier", size=11)

# Header
header = tk.Frame(root, bg="#161b22")
header.pack(fill="x")
tk.Label(header, text="PILLWHEEL  /  SERVO TEST  (ch 0–7)",
         font=f_title, bg="#161b22", fg="#e6edf3",
         padx=14, pady=8).pack(side="left")
hw_txt = "Hardware: ✔ PCA9685" if HW else "Hardware: ⚠ simulation"
tk.Label(header, text=hw_txt, font=f_status, bg="#161b22",
         fg="#3fb950" if HW else "#f85149", padx=14).pack(side="right")

# Button grid
grid = tk.Frame(root, bg="#0d1117")
grid.pack(fill="both", expand=True, padx=20, pady=20)

status_var = tk.StringVar(value="Ready — tap a servo button")

for ch in range(8):
    col = ch % 4
    row = ch // 4
    grid.columnconfigure(col, weight=1)
    grid.rowconfigure(row, weight=1)

    busy = [False]   # mutable flag per button

    btn = tk.Button(
        grid,
        text=f"SERVO {ch}\n(CH {ch:02d})",
        font=f_btn,
        bg=COLOURS[ch],
        fg="white",
        relief="flat",
        cursor="hand2",
        activebackground="#ffffff",
        activeforeground="#000000",
    )

    # Capture btn and ch in closure
    def make_cmd(c, b, bsy):
        def cmd():
            if bsy[0]: return
            status_var.set(f"Rotating ch{c}…")
            b.config(state="disabled", bg="#6e7681")
            rotate(c, status_var, b, bsy)
        return cmd

    btn.config(command=make_cmd(ch, btn, busy))
    btn.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")

# Status bar
tk.Label(root, textvariable=status_var, font=f_status,
         bg="#161b22", fg="#8b949e",
         anchor="w", padx=12, pady=5).pack(side="bottom", fill="x")

# Quit button
tk.Button(root, text="✕  QUIT", font=f_btn,
          bg="#b91c1c", fg="white", relief="flat",
          cursor="hand2", pady=8,
          command=root.quit).pack(side="bottom", fill="x", padx=0)

root.mainloop()






TRASH

"""
Firmware/servo_test_ui.py
Interactive touchscreen UI for testing PillWheel servos individually.

Buttons 0–12 → rotate corresponding dispenser servo (channel 0–12)
Tray Sweep   → run tray_sweep.sweep() on channel 15

Usage:
    export DISPLAY=:0
    python3 firmware/servo_test_ui.py
"""

import sys
import os
import threading
import time

import tkinter as tk
from tkinter import font as tkfont

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from electronic.servo_controller import ServoController
from electronic.tray_sweep import sweep as tray_sweep

# ── Layout ────────────────────────────────────────────────────────────────────
SCREEN_W = 800
SCREEN_H = 480

# Colour palette — dark industrial theme suits a test tool
C_BG       = "#0d1117"
C_PANEL    = "#161b22"
C_BORDER   = "#30363d"
C_IDLE     = "#21262d"
C_ACTIVE   = "#388bfd"
C_ACTIVE_FG= "#ffffff"
C_TRAY     = "#238636"
C_TRAY_FG  = "#ffffff"
C_BUSY     = "#6e7681"
C_FG       = "#e6edf3"
C_MUTED    = "#8b949e"
C_SUCCESS  = "#3fb950"
C_ERROR    = "#f85149"

DISPENSER_COUNT = 13   # channels 0–12


class ServoTestUI:
    """
    Touchscreen test panel.
    Each dispenser button calls ServoController.rotate_dispenser(index).
    The Tray Sweep button calls tray_sweep.sweep().
    Buttons are disabled while a motion is in progress to prevent conflicts.
    """

    def __init__(self, root: tk.Tk):
        self.root  = root
        self.servo = ServoController()
        self._busy = False

        root.title("PillWheel — Servo Test")
        root.geometry(f"{SCREEN_W}x{SCREEN_H}")
        root.configure(bg=C_BG)
        root.bind("<Escape>", lambda e: self._exit())

        self._init_fonts()
        self._build_ui()
        self._set_status("Ready — tap a servo to test it.")

    # ── Fonts ─────────────────────────────────────────────────────────────────

    def _init_fonts(self):
        self.f_title  = tkfont.Font(family="Courier", size=14, weight="bold")
        self.f_btn    = tkfont.Font(family="Courier", size=13, weight="bold")
        self.f_tray   = tkfont.Font(family="Courier", size=14, weight="bold")
        self.f_status = tkfont.Font(family="Courier", size=11)
        self.f_hw     = tkfont.Font(family="Courier", size=10)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=C_PANEL, highlightbackground=C_BORDER,
                          highlightthickness=1)
        header.pack(fill="x")

        tk.Label(header, text="PILLWHEEL  /  SERVO TEST",
                 font=self.f_title, bg=C_PANEL, fg=C_FG,
                 padx=14, pady=8).pack(side="left")

        hw_text = (
            f"Hardware: {'✔ PCA9685' if self.servo.hardware_available else '⚠ simulation'}"
        )
        hw_colour = C_SUCCESS if self.servo.hardware_available else C_ERROR
        tk.Label(header, text=hw_text, font=self.f_hw,
                 bg=C_PANEL, fg=hw_colour, padx=14).pack(side="right")

        # ── Dispenser grid ────────────────────────────────────────────────────
        grid_outer = tk.Frame(self.root, bg=C_BG)
        grid_outer.pack(fill="both", expand=True, padx=14, pady=(10, 4))

        tk.Label(grid_outer, text="DISPENSER SERVOS  (ch 0 – 12)",
                 font=self.f_hw, bg=C_BG, fg=C_MUTED,
                 anchor="w").pack(fill="x", pady=(0, 6))

        grid = tk.Frame(grid_outer, bg=C_BG)
        grid.pack(fill="both", expand=True)

        # 7 columns × 2 rows for channels 0–12  (13 buttons)
        COLS = 7
        self._btns: list[tk.Button] = []

        for idx in range(DISPENSER_COUNT):
            row, col = divmod(idx, COLS)
            btn = tk.Button(
                grid,
                text=f"CH {idx:02d}",
                font=self.f_btn,
                bg=C_IDLE, fg=C_FG,
                activebackground=C_ACTIVE, activeforeground=C_ACTIVE_FG,
                relief="flat", bd=0,
                highlightbackground=C_BORDER, highlightthickness=1,
                cursor="hand2",
                command=lambda i=idx: self._fire_dispenser(i),
            )
            btn.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            grid.rowconfigure(row, weight=1)
            grid.columnconfigure(col, weight=1)
            self._btns.append(btn)

        # ── Tray sweep + quit row ─────────────────────────────────────────────
        tray_frame = tk.Frame(self.root, bg=C_BG)
        tray_frame.pack(fill="x", padx=14, pady=4)

        self._tray_btn = tk.Button(
            tray_frame,
            text="⟳  TRAY SWEEP  (ch 15)",
            font=self.f_tray,
            bg=C_TRAY, fg=C_TRAY_FG,
            activebackground="#2ea043", activeforeground=C_TRAY_FG,
            relief="flat", bd=0,
            highlightbackground=C_BORDER, highlightthickness=1,
            cursor="hand2", pady=10,
            command=self._fire_tray_sweep,
        )
        self._tray_btn.pack(side="left", fill="x", expand=True)

        tk.Button(
            tray_frame,
            text="✕  QUIT",
            font=self.f_tray,
            bg=C_ERROR, fg=C_FG,
            activebackground="#c0392b", activeforeground=C_FG,
            relief="flat", bd=0,
            highlightbackground=C_BORDER, highlightthickness=1,
            cursor="hand2", pady=10, padx=28,
            command=self._exit,
        ).pack(side="left", padx=(8, 0))

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_var = tk.StringVar()
        tk.Label(self.root, textvariable=self._status_var,
                 font=self.f_status, bg=C_PANEL, fg=C_MUTED,
                 anchor="w", padx=12, pady=5,
                 highlightbackground=C_BORDER, highlightthickness=1,
                 ).pack(side="bottom", fill="x")

    # ── Actions ───────────────────────────────────────────────────────────────

    def _fire_dispenser(self, index: int):
        if self._busy:
            return
        self._set_busy(True)
        self._set_status(f"Rotating dispenser servo CH {index:02d}  (0° → 180° → 0°)…")
        self._highlight_btn(index, active=True)

        def _run():
            self.servo.rotate_dispenser(index)
            self.root.after(0, lambda: self._on_done(
                f"CH {index:02d} complete.", index
            ))

        threading.Thread(target=_run, daemon=True).start()

    def _fire_tray_sweep(self):
        if self._busy:
            return
        self._set_busy(True)
        self._set_status("Running tray sweep (ch 15)…")
        self._tray_btn.config(bg="#1a5c27", state="disabled")

        def _run():
            tray_sweep()
            self.root.after(0, self._on_tray_done)

        threading.Thread(target=_run, daemon=True).start()

    # ── Completion callbacks (main thread) ────────────────────────────────────

    def _on_done(self, msg: str, index: int):
        self._highlight_btn(index, active=False)
        self._set_busy(False)
        self._set_status(f"✔  {msg}  — ready.")

    def _on_tray_done(self):
        self._tray_btn.config(bg=C_TRAY, state="normal")
        self._set_busy(False)
        self._set_status("✔  Tray sweep complete — ready.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self._status_var.set(f"  {msg}")

    def _set_busy(self, busy: bool):
        self._busy = busy
        state = "disabled" if busy else "normal"
        bg    = C_BUSY    if busy else C_IDLE
        for btn in self._btns:
            btn.config(state=state, bg=bg)
        self._tray_btn.config(state=state)
        if not busy:
            # Restore tray button colour after non-tray actions
            self._tray_btn.config(bg=C_TRAY)

    def _highlight_btn(self, index: int, active: bool):
        self._btns[index].config(bg=C_ACTIVE if active else C_IDLE)

    def _exit(self):
        self.servo.cleanup()
        self.root.quit()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(True, True)
    app = ServoTestUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.servo.cleanup()