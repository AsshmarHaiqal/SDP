from adafruit_servokit import ServoKit
from time import sleep

kit = ServoKit(channels=16)
kit.servo[9].set_pulse_width_range(500, 2500)

number = 9

def servo():
    for i in range(0, 1):
        kit.servo[number].angle = 0
        sleep(1)
        kit.servo[number].angle = 180
        sleep(1)
        kit.servo[number].angle = 0
        sleep(1)

while True:
    servo()
    sleep(2)