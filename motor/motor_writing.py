import serial

PORT = "COM3"
BAUD = 57600

def send_remocon(ser, value):
    lb = value & 0xFF
    hb = (value >> 8) & 0xFF
    pkt = bytes([0xFF, 0x55, lb, (~lb) & 0xFF, hb, (~hb) & 0xFF])
    ser.write(pkt)

def send_goal(ser, servo_id, ticks):
    value = ((servo_id & 0x0F) << 12) | (ticks & 0x0FFF)
    send_remocon(ser, value)

def send_latch(ser):
    # ID=15 means LATCH
    value = (15 << 12)
    send_remocon(ser, value)

# Example: send 12 positions, then latch
targets = {
    1: 2100, 2: 1900, 3: 2048, 4: 2048,
    5: 2200, 6: 1800, 7: 2048, 8: 2048,
    9: 2300, 10: 1750,11: 2048,12: 2048
}

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    for sid in range(1, 13):
        send_goal(ser, sid, targets[sid])
    send_latch(ser)
