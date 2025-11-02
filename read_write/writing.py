import serial

PORT = "COM3"        # your CM-550 USB COM port
BAUD = 57600

def send_remocon(ser, value):
    lb =  value        & 0xFF
    hb = (value >> 8)  & 0xFF
    pkt = bytes([0xFF, 0x55, lb, (~lb) & 0xFF, hb, (~hb) & 0xFF])
    ser.write(pkt)

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    send_remocon(ser, 2)   # example payload (maps to whatever you handle in CM-550 code)