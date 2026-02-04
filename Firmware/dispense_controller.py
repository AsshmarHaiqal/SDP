import hardware.servo_controller
import hardware.ir_sensor

# Configuration parameters
max_rotates = 5
medicine_dispensed = 0

def dispense_all_medicines():
    global medicine_dispensed

    # Need to import medicine_required from software side
    while medicine_dispensed < medicine_required:
        if dispense_individual_medicine():
            medicine_dispensed += 1
        else:
            return False
    return True

def dispense_individual_medicine():
    number_of_rotates = 0
    successful_dispense = False

    # In hardware/servo_controller.py
    while number_of_rotates < max_rotates and not successful_dispense:
        hardware.servo_controller.rotate_motor()

        # In hardware/ir_sensor.py
        ir_sensor_status = hardware.ir_sensor.retrieve_ir_sensor_status()
    
        if ir_sensor_status == True:
            successful_dispense = True
        else:
            number_of_rotates += 1
    
    return successful_dispense