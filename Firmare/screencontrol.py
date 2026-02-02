
import tkinter as tk
from datetime import datetime

"""
PILL DISPENSER TOUCHSCREEN INTERFACE

Screen Flow:
1. HOME SCREEN: Displays "PillWheel: Automated Medical Dispenser" with current date/time
   - Default state when idle
   - Automatically transitions to "calling patient" when prescription is due

2. CALLING PATIENT SCREEN: Alerts that a patient needs to collect medication
   - Shows "Calling for Patient [ID]"
   - Button: "Ready for Collection" → proceeds to prescription verification

3. PRESCRIPTION VERIFICATION SCREEN: Displays medication details
   - Shows medication name, dosage (e.g., "Paracetamol - 2 pills")
   - Asks "Is this the correct prescription and dose?"
   - Buttons: "Yes" or "No"
   - If "No" → goes to assistance screen
   - If "Yes" → proceeds to dispensing

4. ASSISTANCE SCREEN: If verification failed
   - Shows "Calling for Assistance"
   - Notifies staff that manual intervention is needed
   - Auto-returns to home screen after 10 seconds

5. DISPENSING SCREEN: After successful verification
   - Shows "Pill has been dispensed"
   - Displays dosage instructions (e.g., "Take after eating")
   - Shows next scheduled dose time
   - Button: "Completed" for immediate return to home
   - Auto-returns to home screen after 5 seconds

Logging:
- All actions are logged to the text widget with timestamps
- Log displays at bottom of screen for debugging/monitoring


NOTE: Need to integrate with software team,
Connect to database for real patient/prescription data
Add facial recognition before showing verification screen
Integrate with motor control code to actually dispense pills
Add audio feedback for accessibility
Connect notification system for missed doses
"""

class PillDispenserUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PillWheel Dispenser")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="#f0f0f0")
        
        # Sample prescription data (NEED TO BE LINKED TO DATABASE!!)
        self.current_patient = {
            "id": "001",
            "name": "Patient 1",
            "medication": "Paracetamol",
            "dosage": "2 pills",
            "instructions": "Take after eating",
            "next_dose": "8:00 PM"
        }
        
        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(expand=True, fill="both")
        
        self.log_frame = tk.Frame(root, bg="white", height=150)
        self.log_frame.pack(side="bottom", fill="x")
        self.log_frame.pack_propagate(False)
        
        tk.Label(self.log_frame, text="System Log:", font=("Arial", 12, "bold"), 
                bg="white").pack(anchor="w", padx=10, pady=5)
        
        self.log_text = tk.Text(self.log_frame, font=("Arial", 10), 
                               bg="white", height=6)
        self.log_text.pack(expand=True, fill="both", padx=10, pady=5)
        
        # scrollbar to log
        scrollbar = tk.Scrollbar(self.log_text)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)
        
        self.show_home_screen()
        self.log("System initialized")
        
        self.update_clock()
        
    def log(self, message):
        """Add timestamped message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        
    def clear_main_frame(self):
        """Clear all widgets from main frame"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
    def show_home_screen(self):
        """Display default home screen with branding and time"""
        self.clear_main_frame()
        self.log("Displaying home screen")
        
        # centered container
        container = tk.Frame(self.main_frame, bg="#f0f0f0")
        container.place(relx=0.5, rely=0.5, anchor="center")
        
        # Title
        title = tk.Label(container, 
                        text="PillWheel",
                        font=("Arial", 60, "bold"),
                        fg="#2c3e50",
                        bg="#f0f0f0")
        title.pack(pady=20)
        
        subtitle = tk.Label(container,
                           text="Automated Medical Dispenser",
                           font=("Arial", 28),
                           fg="#34495e",
                           bg="#f0f0f0")
        subtitle.pack(pady=10)
        
        # Date and time display
        self.datetime_label = tk.Label(container,
                                      font=("Arial", 24),
                                      fg="#7f8c8d",
                                      bg="#f0f0f0")
        self.datetime_label.pack(pady=30)
        
        # SIMULATION!!
        test_btn = tk.Button(container,
                           text="Simulate Prescription Due",
                           font=("Arial", 16),
                           bg="#3498db",
                           fg="white",
                           padx=30,
                           pady=15,
                           command=self.show_calling_patient_screen)
        test_btn.pack(pady=30)
        
    def update_clock(self):
        """Update date/time display on home screen"""
        if hasattr(self, 'datetime_label') and self.datetime_label.winfo_exists():
            current_time = datetime.now().strftime("%A, %d %B %Y\n%H:%M:%S")
            self.datetime_label.config(text=current_time)
        self.root.after(1000, self.update_clock)
        
    def show_calling_patient_screen(self):
        """Display screen calling patient to collect medication"""
        self.clear_main_frame()
        self.log(f"Calling patient {self.current_patient['id']}")
        
        container = tk.Frame(self.main_frame, bg="#f0f0f0")
        container.place(relx=0.5, rely=0.5, anchor="center")
        
        # Alert icon (using text, could use image in production)
        icon = tk.Label(container,
                       text="🔔",
                       font=("Arial", 80),
                       bg="#f0f0f0")
        icon.pack(pady=20)
        
        # Message
        message = tk.Label(container,
                          text=f"Calling for {self.current_patient['name']}",
                          font=("Arial", 36, "bold"),
                          fg="#e74c3c",
                          bg="#f0f0f0")
        message.pack(pady=20)
        
        info = tk.Label(container,
                       text="Please approach the dispenser",
                       font=("Arial", 24),
                       fg="#7f8c8d",
                       bg="#f0f0f0")
        info.pack(pady=10)
        
        # Ready button
        ready_btn = tk.Button(container,
                            text="Ready for Collection",
                            font=("Arial", 28, "bold"),
                            bg="#27ae60",
                            fg="white",
                            padx=40,
                            pady=25,
                            command=self.show_verification_screen)
        ready_btn.pack(pady=40)
        
    def show_verification_screen(self):
        """Display prescription details for patient verification"""
        self.clear_main_frame()
        self.log("Showing prescription verification")
        
        container = tk.Frame(self.main_frame, bg="#f0f0f0")
        container.place(relx=0.5, rely=0.5, anchor="center")
        
        header = tk.Label(container,
                         text="Please Verify Your Prescription",
                         font=("Arial", 32, "bold"),
                         fg="#2c3e50",
                         bg="#f0f0f0")
        header.pack(pady=30)
        
        # Prescription details
        details_frame = tk.Frame(container, bg="white", 
                                relief="solid", borderwidth=2)
        details_frame.pack(pady=20, padx=40)
        
        med_name = tk.Label(details_frame,
                           text=f"Medication: {self.current_patient['medication']}",
                           font=("Arial", 28, "bold"),
                           bg="white",
                           fg="#2c3e50")
        med_name.pack(pady=20, padx=50)
        
        dosage = tk.Label(details_frame,
                         text=f"Dosage: {self.current_patient['dosage']}",
                         font=("Arial", 26),
                         bg="white",
                         fg="#34495e")
        dosage.pack(pady=20, padx=50)
        
        # Self-verification on prescription
        question = tk.Label(container,
                           text="Is this the correct prescription and dose?",
                           font=("Arial", 24),
                           fg="#7f8c8d",
                           bg="#f0f0f0")
        question.pack(pady=30)
        
        # Button frame for Yes/No
        button_frame = tk.Frame(container, bg="#f0f0f0")
        button_frame.pack(pady=20)
        
        yes_btn = tk.Button(button_frame,
                          text="YES",
                          font=("Arial", 28, "bold"),
                          bg="#27ae60",
                          fg="white",
                          width=10,
                          padx=30,
                          pady=20,
                          command=self.show_dispensing_screen)
        yes_btn.pack(side="left", padx=20)
        
        no_btn = tk.Button(button_frame,
                         text="NO",
                         font=("Arial", 28, "bold"),
                         bg="#e74c3c",
                         fg="white",
                         width=10,
                         padx=30,
                         pady=20,
                         command=self.show_assistance_screen)
        no_btn.pack(side="left", padx=20)
        
    def show_assistance_screen(self):
        """Display assistance screen when verification fails"""
        self.clear_main_frame()
        self.log("Verification failed - calling for assistance")
        
        container = tk.Frame(self.main_frame, bg="#f0f0f0")
        container.place(relx=0.5, rely=0.5, anchor="center")
        
        # Alert
        icon = tk.Label(container,
                       text="⚠️",
                       font=("Arial", 80),
                       bg="#f0f0f0")
        icon.pack(pady=20)
        
        message = tk.Label(container,
                          text="Calling for Assistance",
                          font=("Arial", 42, "bold"),
                          fg="#e74c3c",
                          bg="#f0f0f0")
        message.pack(pady=20)
        
        info = tk.Label(container,
                       text="A care worker will be with you shortly",
                       font=("Arial", 24),
                       fg="#7f8c8d",
                       bg="#f0f0f0")
        info.pack(pady=10)
        
        # Auto-return to home after 10 seconds
        self.root.after(10000, self.show_home_screen)
        
    def show_dispensing_screen(self):
        """Display dispensing confirmation and instructions"""
        self.clear_main_frame()
        self.log("Dispensing medication")
        
        container = tk.Frame(self.main_frame, bg="#f0f0f0")
        container.place(relx=0.5, rely=0.5, anchor="center")
        
        # Success icon
        icon = tk.Label(container,
                       text="✅",
                       font=("Arial", 80),
                       bg="#f0f0f0")
        icon.pack(pady=20)
        
        # Confirmation
        confirm = tk.Label(container,
                          text="Pill Has Been Dispensed",
                          font=("Arial", 36, "bold"),
                          fg="#27ae60",
                          bg="#f0f0f0")
        confirm.pack(pady=20)
        
        # Instructions box
        instructions_frame = tk.Frame(container, bg="white",
                                     relief="solid", borderwidth=2)
        instructions_frame.pack(pady=20, padx=40)
        
        inst_header = tk.Label(instructions_frame,
                              text="Instructions:",
                              font=("Arial", 24, "bold"),
                              bg="white",
                              fg="#2c3e50")
        inst_header.pack(pady=15, padx=40)
        
        inst_text = tk.Label(instructions_frame,
                            text=self.current_patient['instructions'],
                            font=("Arial", 22),
                            bg="white",
                            fg="#34495e")
        inst_text.pack(pady=15, padx=40)
        
        # Next dose info
        next_dose = tk.Label(container,
                            text=f"Your next dose is scheduled for {self.current_patient['next_dose']}",
                            font=("Arial", 20),
                            fg="#7f8c8d",
                            bg="#f0f0f0")
        next_dose.pack(pady=30)
        
        # Completed button
        complete_btn = tk.Button(container,
                               text="Completed",
                               font=("Arial", 24, "bold"),
                               bg="#3498db",
                               fg="white",
                               padx=40,
                               pady=20,
                               command=self.complete_dispense)
        complete_btn.pack(pady=20)
        
        # Auto-return to home after 5 seconds
        self.root.after(5000, self.complete_dispense)
        
    def complete_dispense(self):
        """Log completion and return to home screen"""
        self.log("Dispensing completed - returning to home screen")
        self.show_home_screen()

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = PillDispenserUI(root)
    root.mainloop()
