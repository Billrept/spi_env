from pycm import *

rc.port(2)   # USB mode
led.set(const.BLUE)

console(const.USB)

print("Ready to receive Remocon packets!")

eeprom.imu_type(const.H)

while True:

    # TEST RECEIVING DATA FROM PC TO CM-550
    if rc.received():
        val = rc.read()
        print("Received:", val)
        if val == 1:
            led.set(const.RED)
        elif val == 2:
            led.set(const.GREEN)
        elif val == 3:
            led.set(const.BLUE)
        elif val == 0:
            led.set()
    
    # TEST SENDING DATA FROM CM-550 TO PC
    ax = imu.accel_x()
    ay = imu.accel_y()
    az = imu.accel_z()
    
    print(ax, ay, az)
    
    delay(200)
