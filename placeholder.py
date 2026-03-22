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