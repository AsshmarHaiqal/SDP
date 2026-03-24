"""
ui/screens.py — tkinter screen builders for PillWheel.

Each build_* function receives the parent Frame and a dict of shared
StringVars / widget references that the app stores and mutates.
"""

import tkinter as tk
from .theme import (
    C_BG, C_PANEL, C_BLUE, C_SUCCESS, C_ERROR, C_WARN, C_WHITE, C_MUTED
)


def build_home(frame: tk.Frame, fonts: dict, on_start, on_maintenance=None):
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=2)
    frame.rowconfigure(1, weight=2)
    frame.rowconfigure(2, weight=1)
    frame.rowconfigure(3, weight=1)

    tk.Label(frame, text="PillWheel", bg=C_BG, fg=C_BLUE,
             font=fonts["title"]).grid(row=0, column=0, pady=(40, 0))

    tk.Button(
        frame, text="Ready to Collect",
        bg=C_BLUE, fg=C_WHITE, font=fonts["btn"],
        relief="flat", cursor="hand2", padx=30, pady=20,
        command=on_start,
    ).grid(row=1, column=0)

    tk.Label(frame, text="Press to begin your medication collection",
             bg=C_BG, fg=C_MUTED, font=fonts["small"]).grid(row=2, column=0)

    # Maintenance button — small, unobtrusive, matching the old main.py layout
    if on_maintenance is not None:
        tk.Button(
            frame, text="Maintenance",
            bg=C_PANEL, fg=C_MUTED, font=fonts["small"],
            relief="flat", cursor="hand2", padx=14, pady=6,
            command=on_maintenance,
        ).grid(row=3, column=0, pady=(0, 10))


def build_scanning(frame: tk.Frame, fonts: dict, on_cancel) -> dict:
    """Returns {'heading': StringVar, 'feed': Label}"""
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=0)
    frame.rowconfigure(1, weight=1)
    frame.rowconfigure(2, weight=0)
    frame.rowconfigure(3, weight=0)

    heading_var = tk.StringVar(value="Scanning for face.")
    tk.Label(frame, textvariable=heading_var, bg=C_BG, fg=C_WHITE,
             font=fonts["title"]).grid(row=0, column=0, pady=(20, 4))

    feed_label = tk.Label(frame, bg=C_PANEL)
    feed_label.grid(row=1, column=0, padx=20, pady=6, sticky="nsew")

    tk.Label(frame, text="Hold still and look at the camera",
             bg=C_BG, fg=C_MUTED, font=fonts["small"]).grid(row=2, column=0, pady=2)

    tk.Button(frame, text="Cancel", bg=C_ERROR, fg=C_WHITE, font=fonts["small"],
              relief="flat", cursor="hand2", padx=20, pady=8,
              command=on_cancel).grid(row=3, column=0, sticky="e", padx=30, pady=10)

    return {"heading": heading_var, "feed": feed_label}


def build_verified(frame: tk.Frame, fonts: dict) -> dict:
    """Returns {'name': StringVar, 'rx': StringVar}"""
    frame.columnconfigure(0, weight=1)
    for r in range(4):
        frame.rowconfigure(r, weight=1)

    name_var = tk.StringVar()
    rx_var   = tk.StringVar()

    tk.Label(frame, text="Identity Confirmed", bg=C_BG, fg=C_SUCCESS,
             font=fonts["title"]).grid(row=0, column=0, pady=(30, 4))
    tk.Label(frame, textvariable=name_var, bg=C_BG, fg=C_WHITE,
             font=fonts["body"]).grid(row=1, column=0)
    tk.Label(frame, textvariable=rx_var, bg=C_BG, fg=C_BLUE,
             font=fonts["body"]).grid(row=2, column=0)
    tk.Label(frame, text="Preparing your medication…", bg=C_BG, fg=C_MUTED,
             font=fonts["small"]).grid(row=3, column=0)

    return {"name": name_var, "rx": rx_var}


def build_dispensing(frame: tk.Frame, fonts: dict) -> dict:
    """Returns {'feed': Label, 'status': StringVar}"""
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=0)
    frame.rowconfigure(1, weight=1)
    frame.rowconfigure(2, weight=0)

    tk.Label(frame, text="Dispensing Medication", bg=C_BG, fg=C_WHITE,
             font=fonts["title"]).grid(row=0, column=0, pady=(20, 4))

    feed_label  = tk.Label(frame, bg=C_PANEL)
    feed_label.grid(row=1, column=0, padx=20, pady=6, sticky="nsew")

    status_var = tk.StringVar()
    tk.Label(frame, textvariable=status_var, bg=C_BG, fg=C_WARN,
             font=fonts["body"]).grid(row=2, column=0, pady=8)

    return {"feed": feed_label, "status": status_var}


def build_complete(frame: tk.Frame, fonts: dict) -> dict:
    """Returns {'name': StringVar, 'details': StringVar, 'cd': StringVar}"""
    frame.columnconfigure(0, weight=1)
    for r in range(4):
        frame.rowconfigure(r, weight=1)

    name_var    = tk.StringVar()
    details_var = tk.StringVar()
    cd_var      = tk.StringVar()

    tk.Label(frame, text="Medication Dispensed", bg=C_BG, fg=C_SUCCESS,
             font=fonts["title"]).grid(row=0, column=0, pady=(30, 4))
    tk.Label(frame, textvariable=name_var, bg=C_BG, fg=C_WHITE,
             font=fonts["body"]).grid(row=1, column=0)
    tk.Label(frame, textvariable=details_var, bg=C_BG, fg=C_BLUE,
             font=fonts["body"]).grid(row=2, column=0)
    tk.Label(frame, textvariable=cd_var, bg=C_BG, fg=C_MUTED,
             font=fonts["small"]).grid(row=3, column=0, pady=4)

    return {"name": name_var, "details": details_var, "cd": cd_var}


def build_error(frame: tk.Frame, fonts: dict) -> dict:
    """Returns {'msg': StringVar, 'cd': StringVar}"""
    frame.columnconfigure(0, weight=1)
    for r in range(4):
        frame.rowconfigure(r, weight=1)

    msg_var = tk.StringVar()
    cd_var  = tk.StringVar()

    tk.Label(frame, text="Something Went Wrong", bg=C_BG, fg=C_ERROR,
             font=fonts["title"]).grid(row=0, column=0, pady=(30, 4))
    tk.Label(frame, textvariable=msg_var, bg=C_BG, fg=C_WHITE,
             font=fonts["body"], wraplength=700, justify="center").grid(
        row=1, column=0, padx=30)
    tk.Label(frame, text="Please call for assistance", bg=C_BG, fg=C_WARN,
             font=fonts["body"]).grid(row=2, column=0)
    tk.Label(frame, textvariable=cd_var, bg=C_BG, fg=C_MUTED,
             font=fonts["small"]).grid(row=3, column=0, pady=4)

    return {"msg": msg_var, "cd": cd_var}