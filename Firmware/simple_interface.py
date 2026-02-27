"""
Firmware/simple_interface.py
Simple two-button touchscreen interface for PillWheel.

Usage (on Raspberry Pi):
    export DISPLAY=:0
    sudo -E python3 Firmware/simple_interface.py
"""

import tkinter as tk
import threading

from electronic.servo_controller import ServoController


# ── Layout ───────────────────────────────────────────────────────────────────

SCREEN_W = 800
SCREEN_H = 480

BG       = "#f0f0f0"
FG       = "#2c3e50"
BTN1_CLR = "#3498db"   # blue  – servo 1
BTN2_CLR = "#9b59b6"   # purple – servo 2
WHITE    = "#ffffff"
DISABLED = "#bdc3c7"


# ── Interface ────────────────────────────────────────────────────────────────

class SimpleInterface:

    def __init__(self, root: tk.Tk, servo: ServoController):
        self.root  = root
        self.servo = servo

        self.root.title("PillWheel")
        self.root.geometry(f"{SCREEN_W}x{SCREEN_H}")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg=BG)
        self.root.bind("<Escape>", lambda e: self._exit())

        self._build_ui()

    def _build_ui(self):
        container = tk.Frame(self.root, bg=BG)
        container.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(container, text="PillWheel",
                 font=("Arial", 52, "bold"), fg=FG, bg=BG).pack(pady=(0, 40))

        btn_frame = tk.Frame(container, bg=BG)
        btn_frame.pack()

        self.btn1 = tk.Button(
            btn_frame, text="Dispense\nServo 1",
            font=("Arial", 24, "bold"), bg=BTN1_CLR, fg=WHITE,
            width=13, height=4, relief="raised", bd=4,
            command=lambda: self._dispense(0, self.btn1),
        )
        self.btn1.pack(side="left", padx=25)

        self.btn2 = tk.Button(
            btn_frame, text="Dispense\nServo 2",
            font=("Arial", 24, "bold"), bg=BTN2_CLR, fg=WHITE,
            width=13, height=4, relief="raised", bd=4,
            command=lambda: self._dispense(1, self.btn2),
        )
        self.btn2.pack(side="left", padx=25)

        tk.Label(container, text="Press ESC to exit",
                 font=("Arial", 12), fg="#95a5a6", bg=BG).pack(pady=(30, 0))

    def _dispense(self, index: int, button: tk.Button):
        """Disable both buttons, rotate the servo in a background thread, then re-enable."""
        self._set_buttons_state("disabled")

        def run():
            self.servo.rotate_dispenser(index)
            self.root.after(0, self._set_buttons_state, "normal")

        threading.Thread(target=run, daemon=True).start()

    def _set_buttons_state(self, state: str):
        colour1 = BTN1_CLR if state == "normal" else DISABLED
        colour2 = BTN2_CLR if state == "normal" else DISABLED
        self.btn1.config(state=state, bg=colour1)
        self.btn2.config(state=state, bg=colour2)

    def _exit(self):
        self.servo.cleanup()
        self.root.quit()


# ── Entry point ──────────────────────────────────────────────────────────────

def simple_interface():
    servo = ServoController()
    root  = tk.Tk()
    SimpleInterface(root, servo)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        servo.cleanup()


if __name__ == "__main__":
    simple_interface()
