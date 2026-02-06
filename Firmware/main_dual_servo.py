#!/usr/bin/env python3
import tkinter as tk
import time

# Import PCA9685 library
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo
    PCA_OK = True
    print("✅ PCA9685 library loaded")
except Exception as e:
    PCA_OK = False
    print(f"⚠️ PCA9685 error: {e}")

# Import sensor
try:
    import adafruit_vl53l0x
    i2c = board.I2C()
    tof = adafruit_vl53l0x.VL53L0X(i2c)
    SENSOR_OK = True
    print("✅ Sensor initialized")
except Exception as e:
    SENSOR_OK = False
    print(f"⚠️ Sensor error: {e}")

# Initialize PCA9685
if PCA_OK:
    try:
        pca = PCA9685(i2c)
        pca.frequency = 50  # 50Hz for servos
        
        # Create servo objects for channels 0 and 1
        servo1 = servo.Servo(pca.channels[0])  # Vitamin D dispenser
        servo2 = servo.Servo(pca.channels[1])  # Vitamin C dispenser
        
        print("✅ Servos initialized on PCA9685")
        print("   Channel 0: Vitamin D dispenser")
        print("   Channel 1: Vitamin C dispenser")
    except Exception as e:
        print(f"⚠️ PCA9685 init error: {e}")
        PCA_OK = False

def get_distance():
    """Read TOF sensor distance"""
    if SENSOR_OK:
        return tof.range
    return 150  # Simulation

def set_servo_angle(servo_obj, angle):
    """Set servo angle 0-180"""
    if PCA_OK:
        servo_obj.angle = angle
        time.sleep(0.5)

def rotate_servo_cycle(servo_obj, servo_name):
    """Rotate servo 0° → 180° → 0° (one dispense cycle)"""
    print(f"   🔄 {servo_name}: 0° → 180° → 0°")
    set_servo_angle(servo_obj, 0)
    time.sleep(0.5)
    set_servo_angle(servo_obj, 180)
    time.sleep(0.5)
    set_servo_angle(servo_obj, 0)
    time.sleep(0.5)

# State variables
numberOfRotates = 0
vitaminD_dispensed = 0
vitaminC_dispensed = 0
baseline_distance = None

# Prescription requirements
VITAMIN_D_REQUIRED = 2
VITAMIN_C_REQUIRED = 1

class PillWheelUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PillWheel - Dual Servo")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="#f0f0f0")
        self.root.bind('<Escape>', lambda e: self.cleanup_and_exit())
        
        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(expand=True, fill="both")
        
        self.current_dispenser = None  # Track which dispenser is active
        self.show_home_screen()
        
    def show_home_screen(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        container = tk.Frame(self.main_frame, bg="#f0f0f0")
        container.place(relx=0.5, rely=0.5, anchor="center")
        
        tk.Label(container, text="PillWheel", font=("Arial", 60, "bold"),
                 fg="#2c3e50", bg="#f0f0f0").pack(pady=20)
        
        tk.Label(container, text="Dual Dispenser System",
                 font=("Arial", 28), fg="#34495e", bg="#f0f0f0").pack(pady=10)
        
        # Status
        status_text = []
        if PCA_OK:
            status_text.append("Servos: OK ✅")
        else:
            status_text.append("Servos: Simulation")
        
        if SENSOR_OK:
            status_text.append("Sensor: OK ✅")
        else:
            status_text.append("Sensor: Simulation")
        
        tk.Label(container, text=" | ".join(status_text), font=("Arial", 14),
                 fg="#27ae60" if (PCA_OK and SENSOR_OK) else "#95a5a6",
                 bg="#f0f0f0").pack(pady=5)
        
        # TEST BUTTONS FRAME
        test_frame = tk.Frame(container, bg="#e8f4f8", relief="solid", borderwidth=2)
        test_frame.pack(pady=20, padx=40)
        
        tk.Label(test_frame, text="Servo Test Controls",
                 font=("Arial", 18, "bold"), bg="#e8f4f8", fg="#2c3e50").pack(pady=10)
        
        # Test button row
        test_btn_frame = tk.Frame(test_frame, bg="#e8f4f8")
        test_btn_frame.pack(pady=10, padx=20)
        
        tk.Button(test_btn_frame, text="Test Servo 1\n(Vitamin D)",
                  font=("Arial", 16), bg="#3498db", fg="white",
                  padx=20, pady=15, command=self.test_servo1).pack(side="left", padx=10)
        
        tk.Button(test_btn_frame, text="Test Servo 2\n(Vitamin C)",
                  font=("Arial", 16), bg="#9b59b6", fg="white",
                  padx=20, pady=15, command=self.test_servo2).pack(side="left", padx=10)
        
        tk.Label(test_frame, text="Each test rotates 0° → 180° → 0°",
                 font=("Arial", 12), bg="#e8f4f8", fg="#7f8c8d").pack(pady=5)
        
        # Main dispense button
        tk.Button(container, text="Start Full Dispense", font=("Arial", 20, "bold"),
                  bg="#27ae60", fg="white", padx=40, pady=20,
                  command=self.show_verification).pack(pady=30)
    
    def test_servo1(self):
        """Test Servo 1 (Vitamin D dispenser)"""
        print("\n" + "="*50)
        print("TESTING SERVO 1 (Vitamin D Dispenser)")
        print("="*50)
        
        if PCA_OK:
            rotate_servo_cycle(servo1, "Servo 1 (Vitamin D)")
            print("✅ Test complete!")
        else:
            print("⚠️ Simulation mode - no hardware")
        
        # Show feedback on screen
        self.show_test_feedback("Servo 1 (Vitamin D)")
    
    def test_servo2(self):
        """Test Servo 2 (Vitamin C dispenser)"""
        print("\n" + "="*50)
        print("TESTING SERVO 2 (Vitamin C Dispenser)")
        print("="*50)
        
        if PCA_OK:
            rotate_servo_cycle(servo2, "Servo 2 (Vitamin C)")
            print("✅ Test complete!")
        else:
            print("⚠️ Simulation mode - no hardware")
        
        # Show feedback on screen
        self.show_test_feedback("Servo 2 (Vitamin C)")
    
    def show_test_feedback(self, servo_name):
        """Show temporary feedback overlay"""
        # Create overlay
        overlay = tk.Frame(self.main_frame, bg="white", relief="solid", borderwidth=3)
        overlay.place(relx=0.5, rely=0.5, anchor="center")
        
        tk.Label(overlay, text="✓", font=("Arial", 60),
                 fg="#27ae60", bg="white").pack(pady=20, padx=60)
        
        tk.Label(overlay, text=f"{servo_name} Test Complete!",
                 font=("Arial", 24, "bold"), bg="white", fg="#2c3e50").pack(pady=10, padx=40)
        
        tk.Label(overlay, text="Servo rotated successfully",
                 font=("Arial", 18), bg="white", fg="#7f8c8d").pack(pady=10, padx=40)
        
        tk.Label(overlay, text=" ", bg="white").pack(pady=10)
        
        # Auto-dismiss after 2 seconds
        self.root.after(2000, overlay.destroy)
                  
    def show_verification(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        container = tk.Frame(self.main_frame, bg="#f0f0f0")
        container.place(relx=0.5, rely=0.5, anchor="center")
        
        tk.Label(container, text="Verify Prescription",
                 font=("Arial", 32, "bold"), fg="#2c3e50", bg="#f0f0f0").pack(pady=30)
        
        # Prescription details box
        details = tk.Frame(container, bg="white", relief="solid", borderwidth=2)
        details.pack(pady=20, padx=40)
        
        tk.Label(details, text="Today's Vitamins:",
                 font=("Arial", 26, "bold"), bg="white", fg="#2c3e50").pack(pady=15, padx=50)
        
        # Vitamin D
        vit_d_frame = tk.Frame(details, bg="white")
        vit_d_frame.pack(pady=10, padx=50)
        tk.Label(vit_d_frame, text="🔸", font=("Arial", 24), bg="white").pack(side="left", padx=5)
        tk.Label(vit_d_frame, text=f"Vitamin D - {VITAMIN_D_REQUIRED} pills",
                 font=("Arial", 24), bg="white", fg="#34495e").pack(side="left")
        
        # Vitamin C
        vit_c_frame = tk.Frame(details, bg="white")
        vit_c_frame.pack(pady=10, padx=50)
        tk.Label(vit_c_frame, text="🔸", font=("Arial", 24), bg="white").pack(side="left", padx=5)
        tk.Label(vit_c_frame, text=f"Vitamin C - {VITAMIN_C_REQUIRED} pill",
                 font=("Arial", 24), bg="white", fg="#34495e").pack(side="left")
        
        tk.Label(details, text=" ", bg="white").pack(pady=5)
        
        tk.Label(container, text="Is this correct?",
                 font=("Arial", 24), fg="#7f8c8d", bg="#f0f0f0").pack(pady=30)
        
        # Buttons
        btn_frame = tk.Frame(container, bg="#f0f0f0")
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="YES", font=("Arial", 28, "bold"),
                  bg="#27ae60", fg="white", width=10, padx=30, pady=20,
                  command=self.start_dispense).pack(side="left", padx=20)
        
        tk.Button(btn_frame, text="NO", font=("Arial", 28, "bold"),
                  bg="#e74c3c", fg="white", width=10, padx=30, pady=20,
                  command=self.call_assistance).pack(side="left", padx=20)
    
    def start_dispense(self):
        global baseline_distance, numberOfRotates, vitaminD_dispensed, vitaminC_dispensed
        
        print("\n" + "="*60)
        print("STARTING DUAL DISPENSE WORKFLOW")
        print("="*60)
        print(f"Target: {VITAMIN_D_REQUIRED}x Vitamin D, {VITAMIN_C_REQUIRED}x Vitamin C")
        
        # Initialize state
        numberOfRotates = 0
        vitaminD_dispensed = 0
        vitaminC_dispensed = 0
        
        # Start with Vitamin D dispenser
        self.current_dispenser = "Vitamin D"
        
        # Measure baseline
        print(f"\nMeasuring baseline distance...")
        baseline_distance = get_distance()
        print(f"✅ Baseline: {baseline_distance:.0f}mm")
        
        self.show_dispensing()
        self.root.after(1000, self.dispense_loop)
    
    def show_dispensing(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        container = tk.Frame(self.main_frame, bg="#f0f0f0")
        container.place(relx=0.5, rely=0.5, anchor="center")
        
        tk.Label(container, text="⏳", font=("Arial", 80), bg="#f0f0f0").pack(pady=20)
        tk.Label(container, text="Dispensing Vitamins...",
                 font=("Arial", 36, "bold"), fg="#3498db", bg="#f0f0f0").pack(pady=20)
        
        self.dispenser_label = tk.Label(container, text="", font=("Arial", 24),
                                        fg="#34495e", bg="#f0f0f0")
        self.dispenser_label.pack(pady=10)
        
        self.status_label = tk.Label(container, text="", font=("Arial", 20),
                                      fg="#7f8c8d", bg="#f0f0f0")
        self.status_label.pack(pady=10)
    
    def dispense_loop(self):
        global numberOfRotates, vitaminD_dispensed, vitaminC_dispensed, baseline_distance
        
        # Check if all done
        if vitaminD_dispensed >= VITAMIN_D_REQUIRED and vitaminC_dispensed >= VITAMIN_C_REQUIRED:
            self.show_success()
            return
        
        # Determine which dispenser to use
        if vitaminD_dispensed < VITAMIN_D_REQUIRED:
            current_servo = servo1
            current_name = "Vitamin D"
            current_count = vitaminD_dispensed
            current_target = VITAMIN_D_REQUIRED
        elif vitaminC_dispensed < VITAMIN_C_REQUIRED:
            current_servo = servo2
            current_name = "Vitamin C"
            current_count = vitaminC_dispensed
            current_target = VITAMIN_C_REQUIRED
        else:
            self.show_success()
            return
        
        # Check max attempts for current pill
        if numberOfRotates >= 5:
            print(f"\n❌ MAX ATTEMPTS for {current_name}")
            self.call_assistance()
            return
        
        # Update UI
        self.dispenser_label.config(text=f"Dispensing: {current_name}")
        self.status_label.config(text=f"Attempt {numberOfRotates + 1}/5 ({current_count + 1}/{current_target})")
        
        numberOfRotates += 1
        
        print(f"\n{'='*60}")
        print(f"{current_name.upper()} - Attempt {numberOfRotates}/5")
        print(f"Progress: {current_count}/{current_target}")
        print(f"{'='*60}")
        
        # Rotate the appropriate servo
        if PCA_OK:
            rotate_servo_cycle(current_servo, current_name)
        else:
            print(f"   🔄 Simulation: {current_name} rotating")
            time.sleep(2)
        
        print("⏳ Waiting for pill to drop...")
        time.sleep(1)
        
        # Check sensor - multiple samples
        print("📏 Detecting pill drop...")
        samples = []
        for i in range(10):
            samples.append(get_distance())
            time.sleep(0.1)
        
        min_dist = min(samples)
        max_dist = max(samples)
        variation = max_dist - min_dist
        
        detected = (abs(min_dist - baseline_distance) >= 5 or 
                   abs(max_dist - baseline_distance) >= 5 or 
                   variation >= 8)
        
        print(f"   Baseline:  {baseline_distance:.0f}mm")
        print(f"   Range:     {min_dist:.0f} - {max_dist:.0f}mm")
        print(f"   Variation: {variation:.0f}mm")
        print(f"   Result:    {'✅ DETECTED' if detected else '❌ NOT DETECTED'}")
        
        if detected:
            # Pill detected - increment the correct counter
            if current_name == "Vitamin D":
                vitaminD_dispensed += 1
                print(f"\n✅ Vitamin D dispensed! Total: {vitaminD_dispensed}/{VITAMIN_D_REQUIRED}")
            else:
                vitaminC_dispensed += 1
                print(f"\n✅ Vitamin C dispensed! Total: {vitaminC_dispensed}/{VITAMIN_C_REQUIRED}")
            
            # Reset for next pill
            numberOfRotates = 0
            time.sleep(1)
            baseline_distance = get_distance()
            print(f"   New baseline: {baseline_distance:.0f}mm")
            
            # Check if switching dispensers
            if vitaminD_dispensed >= VITAMIN_D_REQUIRED and current_name == "Vitamin D":
                print(f"\n✅ All Vitamin D dispensed! Switching to Vitamin C...")
                time.sleep(2)
        else:
            print(f"   Will retry... ({5 - numberOfRotates} attempts remaining)")
        
        self.root.after(500, self.dispense_loop)
    
    def show_success(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        container = tk.Frame(self.main_frame, bg="#f0f0f0")
        container.place(relx=0.5, rely=0.5, anchor="center")
        
        tk.Label(container, text="✅", font=("Arial", 80), bg="#f0f0f0").pack(pady=20)
        tk.Label(container, text="All Vitamins Dispensed!",
                 font=("Arial", 36, "bold"), fg="#27ae60", bg="#f0f0f0").pack(pady=20)
        
        # Summary box
        summary = tk.Frame(container, bg="white", relief="solid", borderwidth=2)
        summary.pack(pady=20, padx=40)
        
        tk.Label(summary, text="Dispensed:", font=("Arial", 24, "bold"),
                 bg="white", fg="#2c3e50").pack(pady=15, padx=40)
        
        tk.Label(summary, text=f"✓ Vitamin D: {vitaminD_dispensed} pills",
                 font=("Arial", 22), bg="white", fg="#27ae60").pack(pady=10, padx=40)
        
        tk.Label(summary, text=f"✓ Vitamin C: {vitaminC_dispensed} pill",
                 font=("Arial", 22), bg="white", fg="#27ae60").pack(pady=10, padx=40)
        
        tk.Label(summary, text=" ", bg="white").pack(pady=5)
        
        tk.Label(container, text="Instructions: Take with food",
                 font=("Arial", 20), fg="#7f8c8d", bg="#f0f0f0").pack(pady=20)
        
        tk.Button(container, text="Complete", font=("Arial", 24, "bold"),
                  bg="#3498db", fg="white", padx=40, pady=20,
                  command=self.show_home_screen).pack(pady=30)
        
        print("\n" + "="*60)
        print("✅ SUCCESS - ALL VITAMINS DISPENSED")
        print(f"   Vitamin D: {vitaminD_dispensed}/{VITAMIN_D_REQUIRED}")
        print(f"   Vitamin C: {vitaminC_dispensed}/{VITAMIN_C_REQUIRED}")
        print("="*60)
    
    def call_assistance(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        container = tk.Frame(self.main_frame, bg="#f0f0f0")
        container.place(relx=0.5, rely=0.5, anchor="center")
        
        tk.Label(container, text="⚠️", font=("Arial", 80), bg="#f0f0f0").pack(pady=20)
        tk.Label(container, text="Calling for Assistance",
                 font=("Arial", 42, "bold"), fg="#e74c3c", bg="#f0f0f0").pack(pady=20)
        
        tk.Label(container, text="A care worker will help you shortly",
                 font=("Arial", 24), fg="#7f8c8d", bg="#f0f0f0").pack(pady=20)
        
        print("\n⚠️ ASSISTANCE CALLED")
        self.root.after(10000, self.show_home_screen)
    
    def cleanup_and_exit(self):
        print("\n🛑 Shutting down...")
        if PCA_OK:
            servo1.angle = 0
            servo2.angle = 0
            pca.deinit()
        print("✅ Cleanup complete")
        self.root.quit()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("        PillWheel Dual Dispenser System")
    print("              Demo 1 - Dual Servo Setup")
    print("="*70)
    print("Hardware:")
    print(f"  Servos:  {'PCA9685 ✅' if PCA_OK else 'Simulation'}")
    print(f"  Sensor:  {'VL53L0X ✅' if SENSOR_OK else 'Simulation'}")
    print("\nPrescription:")
    print(f"  Servo 1 (Ch 0): Vitamin D × {VITAMIN_D_REQUIRED}")
    print(f"  Servo 2 (Ch 1): Vitamin C × {VITAMIN_C_REQUIRED}")
    print("\nPress ESC to exit")
    print("="*70 + "\n")
    
    root = tk.Tk()
    app = PillWheelUI(root)
    root.mainloop()
