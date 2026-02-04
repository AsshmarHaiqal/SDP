from firmware.dispense_controller import DispenseController
from firmware.state_manager import StateManager
from hardware.servo_controller import MotorController
from hardware.ir_sensor import IRSensor
from hardware.display import Display
from config.hardware_config import *
import time

def main():
   # Show ready for collection
   ready_for_collection()

   # Input verification/consent
    verification_consent =bool(input("Press Enter to simulate user verification..."))
    if verification_consent==True:
        # Trigger dispense medicine
        dispense_individual_medicine()
    else:
        call_for_assistance()

   



if __name__ == "__main__":
    main()