
def dispense_individual_medicine():

    # In hardware/servo_controller.py
    trigger_dispense_individual_medicine()
    retrieve_ir_sensor_status()
    if ir_sensor_status == True:
        return True
    else:
        return False