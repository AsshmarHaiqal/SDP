"""
camera_test.py
Live camera preview using tkinter (avoids Qt font issues).

Usage:
    export DISPLAY=:0
    python3 camera_test.py

Press Q or ESC to quit.
"""

import cv2
import tkinter as tk
from PIL import Image, ImageTk

cap = cv2.VideoCapture('/dev/video0')


root = tk.Tk()
root.title("Camera Test — press Q or ESC to quit")
root.configure(bg="#0d1117")
root.bind("<Escape>", lambda e: root.quit())
root.bind("<q>", lambda e: root.quit())
root.bind("<Q>", lambda e: root.quit())

label = tk.Label(root, bg="#0d1117")
label.pack()


def update():
    ret, frame = cap.read()
    if ret:
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        label.config(image=photo)
        label.image = photo
    root.after(30, update)  # ~33 fps


update()
root.mainloop()

cap.release()
print("Camera test done.")