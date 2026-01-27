import tkinter as tk
from datetime import datetime

def dispense():
    log_text.insert(tk.END, f"{datetime.now()} - Dispense button pressed\n")
    log_text.see(tk.END)

app = tk.Tk()
app.title("Pill Dispenser")
app.attributes("-fullscreen", True)

dispense_button = tk.Button(app, text="DISPENSE", font=("Arial", 40), command=dispense)
dispense_button.pack(pady=20)

log_text = tk.Text(app, font=("Arial", 16))
log_text.pack(expand=True, fill="both")

app.mainloop()
