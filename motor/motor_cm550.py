from pycm import *

console(const.USB)
rc.port(2)  # USB remocon channel

print("Remocon multi-servo receiver (buffer + latch)")

# Power the bus and prep 12 motors (IDs 1..12) in position mode
dxlbus.power_on()
MOT_IDS = list(range(1, 13))
mot = {sid: DXL(sid) for sid in MOT_IDS}
for sid in MOT_IDS:
    m = mot[sid]
    m.mode(3)          # position mode
    m.torque_on()

# Buffer of next goals (ticks)
buf = {sid: 2048 for sid in MOT_IDS}  # default center
HAVE = set()  # which IDs got updated in current frame

def apply_latch():
    # Apply all buffered goals “together”
    for sid in MOT_IDS:
        mot[sid].goal_position(buf[sid])

while True:
    if rc.received():
        v = rc.read()              # 16-bit
        sid = (v >> 12) & 0x0F     # upper 4 bits
        pos = v & 0x0FFF           # lower 12 bits (0..4095)

        if sid == 15:
            # LATCH: push all buffered goals now
            apply_latch()
            print("LATCH applied; frame = { " + ", ".join(["%d:%d" % (i, buf[i]) for i in MOT_IDS]) + " }")
            HAVE.clear()
        elif 1 <= sid <= 12:
            buf[sid] = pos
            HAVE.add(sid)
            # Optional: feedback for debugging
            # print(f"Buffered ID {sid} -> {pos}")
        else:
            # ignore invalid id
            pass

    delay(2)
