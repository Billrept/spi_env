from pycm import *
import math

rc.port(2)   # USB mode
led.set(const.BLUE)

console(const.USB)
print("Kalman IMU filter (scaled) starting...")

eeprom.imu_type(const.H)

# -------- Kalman for one axis (unchanged) --------
class KalmanAxis:
    def __init__(self, Q_angle=0.001, Q_bias=0.003, R_measure=0.03):
        self.Q_angle = Q_angle
        self.Q_bias  = Q_bias
        self.R_measure = R_measure
        self.angle = 0.0
        self.bias  = 0.0
        self.P00 = self.P01 = self.P10 = self.P11 = 0.0
    def set_angle(self, angle): self.angle = angle
    def update(self, meas_angle_deg, gyro_rate_dps, dt):
        rate = gyro_rate_dps - self.bias
        self.angle += dt * rate
        self.P00 += dt*(dt*self.P11 - self.P01 - self.P10 + self.Q_angle)
        self.P01 += -dt*self.P11
        self.P10 += -dt*self.P11
        self.P11 += self.Q_bias * dt
        S = self.P00 + self.R_measure
        K0 = self.P00 / S; K1 = self.P10 / S
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
    # ax_g, ay_g, az_g in g
    roll_deg  = math.degrees(math.atan2(ay_g, az_g)) if (ay_g*ay_g+az_g*az_g)>1e-9 else 0.0
    denom = math.sqrt(ay_g*ay_g + az_g*az_g)
    pitch_deg = math.degrees(math.atan2(-ax_g, denom)) if denom>1e-9 else 0.0
    return roll_deg, pitch_deg

# ----- Gyro bias calibration (use scaled units) -----
print("Calibrating gyro bias... keep still")
N = 200
gx_b = 0.0; gy_b = 0.0
for _ in range(N):
    gx_b += imu.gyro_x() / 100.0   # centi-deg/s → deg/s
    gy_b += imu.gyro_y() / 100.0
    delay(5)
gx_b /= N; gy_b /= N
print("Gyro bias (deg/s):", gx_b, gy_b)

# ----- Init filters with accel-only angle (scaled) -----
ax = imu.accel_x()/1000.0; ay = imu.accel_y()/1000.0; az = imu.accel_z()/1000.0  # mg → g
roll0, pitch0 = accel_to_roll_pitch_deg(ax, ay, az)
kf_roll  = KalmanAxis();  kf_roll.set_angle(roll0)
kf_pitch = KalmanAxis();  kf_pitch.set_angle(pitch0)

t_prev = millis()

while True:
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
    
    t_now = millis()
    dt = max(0.001, min(0.05, (t_now - t_prev)/1000.0))
    t_prev = t_now

    # SCALE EVERYTHING TO PHYSICAL UNITS
    ax = imu.accel_x()/1000.0; ay = imu.accel_y()/1000.0; az = imu.accel_z()/1000.0  # g
    gx = imu.gyro_x()/100.0 - gx_b   # deg/s
    gy = imu.gyro_y()/100.0 - gy_b   # deg/s

    roll_acc,  pitch_acc  = accel_to_roll_pitch_deg(ax, ay, az)  # deg
    roll_kf  = kf_roll.update(roll_acc,  gx, dt)                 # deg
    pitch_kf = kf_pitch.update(pitch_acc, gy, dt)                 # deg

    # (Optional) yaw in centi-deg → deg if you want to print it:
    yaw_deg = imu.yaw() / 100.0

    print("RAW  R/P/Y(deg): {:.2f} {:.2f} {:.2f} | ACC(g): {:.3f} {:.3f} {:.3f} | GYRO(dps): {:.2f} {:.2f}"
          .format(imu.roll()/100.0, imu.pitch()/100.0, yaw_deg, ax, ay, az, gx, gy))
    print("KF   R/P(deg):   {:.2f} {:.2f}".format(roll_kf, pitch_kf))
    delay(20)  # ~50 Hz
