"""
Firmware/sound_test_ui.py
Touchscreen test panel for SoundActuator / TTS.

Each button triggers one of the named phrases from PHRASES.
A custom text field lets us type and speak anything.
A status bar shows which engine is active and what was last spoken.

Usage:
    export DISPLAY=:0
    python3 firmware/sound_test_ui.py
"""

import sys
import os
import threading

import tkinter as tk
from tkinter import font as tkfont

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from electronic.sound_actuator import SoundActuator, TTS_MODEL, PHRASES

# ── Dimensions ────────────────────────────────────────────────────────────────
W, H = 800, 480

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG     = "#0d1117"
C_PANEL  = "#161b22"
C_BORDER = "#30363d"
C_FG     = "#e6edf3"
C_MUTED  = "#8b949e"
C_IDLE   = "#21262d"
C_BUSY   = "#2d333b"
C_SPEAK  = "#388bfd"       # blue  — phrase buttons (speaking)
C_CUSTOM = "#238636"       # green — custom speak button
C_QUIT   = "#b91c1c"       # red   — quit
C_ACTIVE = "#58a6ff"       # highlight while audio is playing

# Each phrase key gets a friendly label and a colour group
_PHRASE_META: dict[str, tuple[str, str]] = {
    "ready":           ("Ready for Collection",   "#1f6feb"),
    "verifying":       ("Verifying Face",          "#1f6feb"),
    "verified":        ("Identity Confirmed",      "#238636"),
    "access_denied":   ("Access Denied",           "#b91c1c"),
    "dispensing":      ("Dispensing",              "#1f6feb"),
    "take_with_food":  ("Take with Food",          "#9e6a03"),
    "collected":       ("Medication Collected",    "#238636"),
    "error":           ("Error / Malfunction",     "#b91c1c"),
    "missed_dose":     ("Missed Dose Alert",       "#b91c1c"),
    "low_stock":       ("Low Stock Warning",       "#9e6a03"),
    "no_prescription": ("No Prescription Found",   "#9e6a03"),
    "count_mismatch":  ("Count Mismatch",          "#b91c1c"),
}

_ENGINE_NAMES = {1: "Piper TTS (local neural)", 2: "Edge TTS (online)", 3: "eSpeak (fallback)"}


class SoundTestUI:

    def __init__(self, root: tk.Tk):
        self.root   = root
        self.sound  = SoundActuator()
        self._busy  = False

        root.title("PillWheel — Sound Actuator Test")
        root.geometry(f"{W}x{H}")
        root.configure(bg=C_BG)
        root.resizable(True, True)
        root.bind("<Escape>", lambda e: self._quit())

        self._init_fonts()
        self._build_ui()
        self._set_status(f"Ready — engine: {_ENGINE_NAMES.get(TTS_MODEL, 'unknown')}")

    # ── Fonts ─────────────────────────────────────────────────────────────────

    def _init_fonts(self):
        self.f_title  = tkfont.Font(family="Courier", size=13, weight="bold")
        self.f_btn    = tkfont.Font(family="Courier", size=11, weight="bold")
        self.f_small  = tkfont.Font(family="Courier", size=10)
        self.f_status = tkfont.Font(family="Courier", size=10)
        self.f_entry  = tkfont.Font(family="Courier", size=12)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=C_PANEL,
                          highlightbackground=C_BORDER, highlightthickness=1)
        header.pack(fill="x")

        tk.Label(header, text="PILLWHEEL  /  SOUND TEST",
                 font=self.f_title, bg=C_PANEL, fg=C_FG,
                 padx=14, pady=7).pack(side="left")

        engine_label = _ENGINE_NAMES.get(TTS_MODEL, f"Model {TTS_MODEL}")
        tk.Label(header, text=f"Engine: {engine_label}",
                 font=self.f_small, bg=C_PANEL, fg=C_MUTED, padx=14).pack(side="right")

        # ── Phrase buttons grid ───────────────────────────────────────────────
        grid_outer = tk.Frame(self.root, bg=C_BG)
        grid_outer.pack(fill="both", expand=True, padx=12, pady=(8, 2))

        tk.Label(grid_outer, text="SYSTEM PHRASES",
                 font=self.f_small, bg=C_BG, fg=C_MUTED,
                 anchor="w").pack(fill="x", pady=(0, 4))

        grid = tk.Frame(grid_outer, bg=C_BG)
        grid.pack(fill="both", expand=True)

        COLS = 4
        self._phrase_btns: dict[str, tk.Button] = {}

        for i, (key, (label, colour)) in enumerate(_PHRASE_META.items()):
            row, col = divmod(i, COLS)
            btn = tk.Button(
                grid,
                text=label,
                font=self.f_btn,
                bg=C_IDLE, fg=C_FG,
                activebackground=colour, activeforeground=C_FG,
                relief="flat", bd=0,
                highlightbackground=C_BORDER, highlightthickness=1,
                cursor="hand2", wraplength=160, justify="center",
                pady=8,
                command=lambda k=key, c=colour: self._speak_phrase(k, c),
            )
            btn.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
            grid.rowconfigure(row, weight=1)
            grid.columnconfigure(col, weight=1)
            self._phrase_btns[key] = btn

        # ── Custom text row ───────────────────────────────────────────────────
        custom_frame = tk.Frame(self.root, bg=C_BG)
        custom_frame.pack(fill="x", padx=12, pady=(4, 2))

        tk.Label(custom_frame, text="CUSTOM:",
                 font=self.f_btn, bg=C_BG, fg=C_MUTED,
                 padx=6).pack(side="left")

        self._custom_var = tk.StringVar()
        self._entry = tk.Entry(
            custom_frame,
            textvariable=self._custom_var,
            font=self.f_entry,
            bg=C_IDLE, fg=C_FG,
            insertbackground=C_FG,
            relief="flat",
            highlightbackground=C_BORDER, highlightthickness=1,
        )
        self._entry.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=6)
        self._entry.bind("<Return>", lambda e: self._speak_custom())

        self._speak_btn = tk.Button(
            custom_frame,
            text="▶ SPEAK",
            font=self.f_btn, bg=C_CUSTOM, fg=C_FG,
            activebackground="#166534", relief="flat", cursor="hand2",
            padx=14, pady=6,
            command=self._speak_custom,
        )
        self._speak_btn.pack(side="left")

        self._quit_btn = tk.Button(
            custom_frame,
            text="✕ QUIT",
            font=self.f_btn, bg=C_QUIT, fg=C_FG,
            activebackground="#991b1b", relief="flat", cursor="hand2",
            padx=14, pady=6,
            command=self._quit,
        )
        self._quit_btn.pack(side="left", padx=(8, 0))

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_var = tk.StringVar()
        tk.Label(
            self.root, textvariable=self._status_var,
            font=self.f_status, bg=C_PANEL, fg=C_MUTED,
            anchor="w", padx=12, pady=4,
            highlightbackground=C_BORDER, highlightthickness=1,
        ).pack(side="bottom", fill="x")

    # ── Actions ───────────────────────────────────────────────────────────────

    def _speak_phrase(self, key: str, colour: str):
        if self._busy:
            return
        text = PHRASES[key]
        label, _ = _PHRASE_META[key]
        self._set_busy(True)
        self._set_status(f'Speaking: "{label}"')
        self._phrase_btns[key].config(bg=colour)

        def _run():
            self.sound.speak_wait(text)
            self.root.after(0, lambda: self._on_done(key))

        threading.Thread(target=_run, daemon=True).start()

    def _speak_custom(self):
        text = self._custom_var.get().strip()
        if not text or self._busy:
            return
        self._set_busy(True)
        self._set_status(f'Speaking custom: "{text[:60]}{"…" if len(text) > 60 else ""}"')
        self._speak_btn.config(bg="#1a5c27")

        def _run():
            self.sound.speak_wait(text)
            self.root.after(0, self._on_custom_done)

        threading.Thread(target=_run, daemon=True).start()

    # ── Completion callbacks ───────────────────────────────────────────────────

    def _on_done(self, key: str):
        self._phrase_btns[key].config(bg=C_IDLE)
        self._set_busy(False)
        self._set_status(f"Done. Engine: {_ENGINE_NAMES.get(TTS_MODEL, 'unknown')}")

    def _on_custom_done(self):
        self._speak_btn.config(bg=C_CUSTOM)
        self._set_busy(False)
        self._set_status(f"Done. Engine: {_ENGINE_NAMES.get(TTS_MODEL, 'unknown')}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self._status_var.set(f"  {msg}")

    def _set_busy(self, busy: bool):
        self._busy = busy
        state = "disabled" if busy else "normal"
        for btn in self._phrase_btns.values():
            btn.config(state=state)
        self._speak_btn.config(state=state)
        self._entry.config(state=state)

    def _quit(self):
        self.root.quit()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app  = SoundTestUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass