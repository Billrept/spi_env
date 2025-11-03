from pycm import *
import math

rc.port(2)
led.set(const.BLUE)
console(const.USB)
print("RL Spider Controller starting...")
eeprom.imu_type(const.H)

# ========== SERVO SETUP ==========
# Assuming 18 servos (3 DOF × 6 legs) - adjust to your spider config
SERVO_IDS = list(range(1, 19))  # IDs 1-18
NUM_SERVOS = len(SERVO_IDS)

# Initialize servos
for sid in SERVO_IDS:
    dxl.torque(sid, 1)  # Enable torque
    dxl.goal(sid, 512)  # Center position (0-1023 range)

# Servo position limits (safety)
SERVO_MIN = 200
SERVO_MAX = 824

# ========== KALMAN FILTER (your existing code) ==========
class KalmanAxis:
    def __init__(self, Q_angle=0.001, Q_bias=0.003, R_measure=0.03):
        self.Q_angle = Q_angle
        self.Q_bias  = Q_bias
        self.R_measure = R_measure
        self.angle = 0.0
        self.bias  = 0.0
        self.P00 = self.P01 = self.P10 = self.P11 = 0.0
    
    def set_angle(self, angle): 
        self.angle = angle
    
    def update(self, meas_angle_deg, gyro_rate_dps, dt):
        rate = gyro_rate_dps - self.bias
        self.angle += dt * rate
        self.P00 += dt*(dt*self.P11 - self.P01 - self.P10 + self.Q_angle)
        self.P01 += -dt*self.P11
        self.P10 += -dt*self.P11
        self.P11 += self.Q_bias * dt
        S = self.P00 + self.R_measure
        K0 = self.P00 / S
        K1 = self.P10 / S
        y = meas_angle_deg - self.angle
        self.angle += K0 * y
        self.bias  += K1 * y
        P00 = self.P00 - K0*self.P00
        P01 = self.P01 - K0*self.P01
        P10 = self.P10 - K1*self.P00
        P11 = self.P11 - K1*self.P01
        self.P00, self.P01, self.P10, self.P11 = P00, P01, P10, P11
        return self.angle

def accel_to_roll_pitch_deg(ax_g, ay_g, az_g):
    roll_deg  = math.degrees(math.atan2(ay_g, az_g)) if (ay_g*ay_g+az_g*az_g)>1e-9 else 0.0
    denom = math.sqrt(ay_g*ay_g + az_g*az_g)
    pitch_deg = math.degrees(math.atan2(-ax_g, denom)) if denom>1e-9 else 0.0
    return roll_deg, pitch_deg

# ========== GYRO CALIBRATION (your existing code) ==========
print("Calibrating gyro bias... keep still")
N = 200
gx_b = 0.0
gy_b = 0.0
for _ in range(N):
    gx_b += imu.gyro_x() / 100.0
    gy_b += imu.gyro_y() / 100.0
    delay(5)
gx_b /= N
gy_b /= N
print("Gyro bias (deg/s):", gx_b, gy_b)

# ========== INIT KALMAN FILTERS ==========
ax = imu.accel_x()/1000.0
ay = imu.accel_y()/1000.0
az = imu.accel_z()/1000.0
roll0, pitch0 = accel_to_roll_pitch_deg(ax, ay, az)
kf_roll  = KalmanAxis()
kf_roll.set_angle(roll0)
kf_pitch = KalmanAxis()
kf_pitch.set_angle(pitch0)

# ========== STATE & ACTION FUNCTIONS ==========
def get_state():
    """
    Read all sensors and return state vector
    State: [joint_positions (18), joint_velocities (18), IMU (6)] = 42 dims
    """
    state = []
    
    # Joint positions (normalized to [-1, 1])
    for sid in SERVO_IDS:
        pos = dxl.position(sid)  # 0-1023
        pos_norm = (pos - 512) / 512.0  # Normalize around center
        state.append(pos_norm)
    
    # Joint velocities (normalized)
    for sid in SERVO_IDS:
        vel = dxl.speed(sid)  # Read velocity
        vel_norm = vel / 1023.0  # Normalize
        state.append(vel_norm)
    
    # IMU data (already in physical units)
    ax = imu.accel_x()/1000.0
    ay = imu.accel_y()/1000.0
    az = imu.accel_z()/1000.0
    gx = (imu.gyro_x()/100.0 - gx_b) / 100.0  # Normalize to reasonable range
    gy = (imu.gyro_y()/100.0 - gy_b) / 100.0
    
    roll_acc, pitch_acc = accel_to_roll_pitch_deg(ax, ay, az)
    
    state.extend([roll_acc/90.0, pitch_acc/90.0, ax, ay, az, gx, gy])  # Normalize angles
    
    return state

def set_action(action):
    """
    Execute action on servos
    Action: [18] joint target positions in [-1, 1]
    """
    for i, sid in enumerate(SERVO_IDS):
        if i < len(action):
            # Convert from [-1, 1] to servo range
            target = int(512 + action[i] * 312)  # Center ± range
            target = max(SERVO_MIN, min(SERVO_MAX, target))  # Safety clamp
            dxl.goal(sid, target)

def check_termination():
    """
    Check if episode should end (robot fell)
    """
    ax = imu.accel_x()/1000.0
    ay = imu.accel_y()/1000.0
    az = imu.accel_z()/1000.0
    roll_acc, pitch_acc = accel_to_roll_pitch_deg(ax, ay, az)
    
    # Terminate if tilted too much
    if abs(roll_acc) > 45 or abs(pitch_acc) > 45:
        return True
    return False

# ========== PROTOCOL FOR HOST COMMUNICATION ==========
"""
Commands from host computer (via Serial/RC):
- 'S' → Send current state
- 'A,v1,v2,...,v18' → Set action (18 comma-separated values)
- 'R' → Reset robot to neutral position
- 'T' → Check termination
"""

def parse_command(cmd_str):
    """Parse command from host"""
    if cmd_str == 'S':
        # Send state
        state = get_state()
        print("STATE," + ",".join(["{:.4f}".format(x) for x in state]))
    
    elif cmd_str.startswith('A,'):
        # Set action
        parts = cmd_str.split(',')
        action = [float(x) for x in parts[1:]]
        set_action(action)
        print("ACK")
    
    elif cmd_str == 'R':
        # Reset to neutral
        for sid in SERVO_IDS:
            dxl.goal(sid, 512)
        delay(1000)
        print("RESET_DONE")
    
    elif cmd_str == 'T':
        # Check termination
        done = check_termination()
        print("TERM," + str(int(done)))
    
    else:
        print("UNKNOWN")

# ========== MAIN LOOP ==========
print("Ready for RL control")
led.set(const.GREEN)

cmd_buffer = ""
while True:
    # Read commands from serial
    if rc.received():
        char = chr(rc.read())
        if char == '':
            parse_command(cmd_buffer.strip())
            cmd_buffer = ""
        else:
            cmd_buffer += char
    
    delay(10)  # 100 Hz control loop