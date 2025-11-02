import serial, threading, time, math, re, numpy as np
from typing import Optional, List

# ===== Serial (edit these) =====
PORT = "COM3"
BAUD = 57600
TIMEOUT_S = 0.05

# ===== Robot constants =====
TICKS_PER_REV = 4096
TICKS2RAD = 2 * math.pi / TICKS_PER_REV
RAD2TICKS = 1.0 / TICKS2RAD

# Real joint limits (radians). First 6 = H, next 6 = V. Adjust per your mechanics.
JOINT_LIMITS_H = [(-2.094, 2.094)] * 6  # ±120°
JOINT_LIMITS_V = [(-2.094, 2.094)] * 6
JOINT_LIMITS_12 = JOINT_LIMITS_H + JOINT_LIMITS_V

# Channel mapping: which channel id (0..11) controls which joint index.
# By default: H joints -> channels 0..5, V joints -> channels 6..11.
CH_MAP_H = [0, 1, 2, 3, 4, 5]
CH_MAP_V = [6, 7, 8, 9, 10, 11]
CH_MAP_12 = CH_MAP_H + CH_MAP_V

# ===== Remocon framing =====
# value16 = [chan:4 bits][ticks:12 bits]; packet = FF 55 LB ~LB HB ~HB
def _checksum_byte(x): return (~x) & 0xFF
def _pack_remocon(value16: int) -> bytes:
    lb = value16 & 0xFF
    hb = (value16 >> 8) & 0xFF
    return bytes([0xFF, 0x55, lb, _checksum_byte(lb), hb, _checksum_byte(hb)])

def _ticks_from_rad(q: float) -> int:
    return int(np.clip(round(q * RAD2TICKS), 0, 4095))

DEG2RAD = math.pi / 180.0

class Remocon12Link:
    """Serial link:
       - Reads IMU from ASCII lines like:
         '[RX] RAW  R/P/Y(deg): 0.39 -1.54 14.24 | ACC(g): 0.029 0.006 0.967 | GYRO(dps): -0.05 0.01'
         '[RX] KF   R/P(deg):   0.43 -1.58'
       - Sends 12 motor targets via Remocon packets (one packet per joint).
    """
    def __init__(self, port: str = PORT, baud: int = BAUD, timeout: float = TIMEOUT_S):
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        self._rx_thread = threading.Thread(target=self._reader, daemon=True)
        self._lock = threading.Lock()
        self._last_imu = None   # dict with rpy(rad), gyro(rad/s), acc(g)
        self._running = True
        self._rx_thread.start()

    # ---------- writer: 12 joint angles (radians) ----------
    def send_joint_targets_rad12(self, q_rad12: List[float]):
        for joint_idx, q in enumerate(q_rad12[:12]):
            lo, hi = JOINT_LIMITS_12[joint_idx]
            q = float(np.clip(q, lo, hi))
            ticks = _ticks_from_rad(q) & 0x0FFF
            channel = CH_MAP_12[joint_idx] & 0x0F
            value = (channel << 12) | ticks
            self.ser.write(_pack_remocon(value))
            # 12 packets @ 57.6kbps is tight; tiny pacing helps USB stacks
            time.sleep(0.0008)

    # ---------- reader: parse RAW / KF lines ----------
    def _parse_raw_line(self, line: str) -> Optional[dict]:
        try:
            rpy = re.search(r'R/P/Y\(deg\):\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', line)
            acc = re.search(r'ACC\(g\):\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', line)
            gy  = re.search(r'GYRO\(dps\):\s*([-\d.]+)\s+([-\d.]+)(?:\s+([-\d.]+))?', line)
            if not (rpy and acc):
                return None
            roll_d, pitch_d, yaw_d = map(float, rpy.groups())
            ax_g, ay_g, az_g       = map(float, acc.groups())
            if gy:
                gx = float(gy.group(1)); gy_ = float(gy.group(2))
                gz = float(gy.group(3)) if gy.group(3) is not None else 0.0
            else:
                gx = gy_ = gz = 0.0
            return {
                "rpy":  (roll_d*DEG2RAD, pitch_d*DEG2RAD, yaw_d*DEG2RAD),
                "gyro": (gx*DEG2RAD,     gy_*DEG2RAD,     gz*DEG2RAD),
                "acc":  (ax_g, ay_g, az_g)
            }
        except Exception:
            return None

    def _parse_kf_line(self, line: str):
        try:
            m = re.search(r'KF\s+R/P\(deg\):\s*([-\d.]+)\s+([-\d.]+)', line)
            if not m: return None
            return (float(m.group(1))*DEG2RAD, float(m.group(2))*DEG2RAD)
        except Exception:
            return None

    def _reader(self):
        kf_rp = None
        while self._running:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
                if not line: 
                    continue
                if "RAW" in line:
                    raw = self._parse_raw_line(line)
                    if raw:
                        if kf_rp:
                            r, p = kf_rp
                            raw["rpy"] = (r, p, raw["rpy"][2])
                        with self._lock:
                            self._last_imu = raw
                elif "KF" in line:
                    kf_rp = self._parse_kf_line(line) or kf_rp
            except Exception:
                time.sleep(0.01)

    def wait_first_imu(self) -> dict:
        while True:
            with self._lock:
                if self._last_imu is not None:
                    return self._last_imu
            time.sleep(0.01)

    def read_latest_imu(self, max_wait_s: float) -> Optional[dict]:
        t0 = time.time()
        while time.time() - t0 < max_wait_s:
            with self._lock:
                if self._last_imu is not None:
                    return self._last_imu
            time.sleep(0.003)
        with self._lock:
            return self._last_imu

    def close(self):
        self._running = False
        try: self.ser.close()
        except Exception: pass