# FIXED VERSION - Key changes marked with # FIX comments

import gymnasium as gym
import numpy as np, math, time, random
from gymnasium import spaces

CTRL_HZ = 50
DT = 1.0 / CTRL_HZ
EPISODE_SECONDS = 10.0
FALL_DEG = 60.0

MAX_STEP_RAD_H = 0.1  # FIX: Increased from 0.05 for more dynamic movement
MAX_STEP_RAD_V = 0.08  # FIX: Increased from 0.05

JOINT_LIMITS_12 = [(-0.785, 0.785)] * 12

class SPIBalance12Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, port=None, use_hardware: bool=False, seed: int | None=None,
                 episode_seconds: float=None, max_tilt_deg: float=None):
        super().__init__()
        self.use_hardware = use_hardware
        self.port = port
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        self.episode_seconds = episode_seconds if episode_seconds is not None else EPISODE_SECONDS
        self.fall_deg = max_tilt_deg if max_tilt_deg is not None else FALL_DEG
        
        self._state = np.zeros(6, dtype=np.float32)
        self._acc   = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        # FIX: Explicitly initialize velocity
        self._velocity_x = 0.0
        self._position_x = 0.0
        self._last_position_x = 0.0
        
        # FIX: Track actual forward movement for better reward calculation
        self._step_count = 0
        self._cumulative_distance = 0.0
        
        high_imu = np.array([math.pi]*3 + [50.0]*3 + [5.0]*3, dtype=np.float32)
        high_q   = np.array([0.785]*12, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.concatenate([high_imu, high_q]),
            high=np.concatenate([high_imu, high_q]),
            dtype=np.float32
        )

        low  = np.array([-MAX_STEP_RAD_H]*6 + [-MAX_STEP_RAD_V]*6, dtype=np.float32)
        high = np.array([+MAX_STEP_RAD_H]*6 + [+MAX_STEP_RAD_V]*6, dtype=np.float32)
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        self._theta_cmd = np.zeros(12, dtype=np.float32)
        self._t = 0
        self.steps_max = int(self.episode_seconds * CTRL_HZ)

        self.link = None
        if self.use_hardware:
            from textimu_link import Remocon12Link
            self.link = Remocon12Link(port=self.port)

    def _imu_offline(self):
        """FIXED: Simplified and more effective physics"""
        x = self._state.copy()
        qH = self._theta_cmd[:6]
        qV = self._theta_cmd[6:]

        # FIX: Much simpler and more effective forward force calculation
        # Key insight: Treat hexapod like a differential drive
        # Left legs (0,2,4) vs Right legs (1,3,5)
        
        left_sweep = np.mean(qH[[0, 2, 4]])   # Average left leg angles
        right_sweep = np.mean(qH[[1, 3, 5]])  # Average right leg angles
        
        # Forward motion when legs sweep backward (negative for left, positive for right)
        # This creates a "rowing" motion
        forward_push = -(left_sweep + right_sweep) / 2.0  # Both pushing back = forward
        
        # Vertical joints control ground contact (negative = pushing down = good)
        ground_contact = np.mean(np.clip(-qV, 0.0, 0.785))  # Only count downward push
        
        # FIX: Combined force with much stronger coefficient
        # Scale by 10.0 to make movement clearly visible
        forward_force = forward_push * (0.5 + ground_contact) * 10.0
        
        # FIX: Simplified dynamics
        MASS = 1.2  # Lighter
        FRICTION = 0.9  # High grip
        DRAG = 0.85  # Some resistance (lower = more drag)
        
        # Update velocity
        forward_accel = (forward_force * FRICTION) / MASS
        self._velocity_x = self._velocity_x * DRAG + forward_accel * DT
        self._velocity_x = np.clip(self._velocity_x, -0.5, 1.0)  # Realistic limits
        
        # Update position
        self._position_x += self._velocity_x * DT
        
        # Balance dynamics (simplified)
        left_legs_v = qV[[0, 2, 4]]
        right_legs_v = qV[[1, 3, 5]]
        front_legs_v = qV[[0, 1]]
        rear_legs_v = qV[[4, 5]]

        r_cmd = 0.4 * (np.mean(right_legs_v) - np.mean(left_legs_v))
        p_cmd = 0.4 * (np.mean(front_legs_v) - np.mean(rear_legs_v))
        y_cmd = 0.2 * (np.mean(qH[[0, 2, 4]]) - np.mean(qH[[1, 3, 5]]))

        u = np.array([r_cmd, p_cmd, y_cmd], dtype=np.float32)

        zeta = 0.7
        wn = 6.0
        Apos = np.array([[-2*zeta*wn, 0, 0],
                        [0, -2*zeta*wn, 0],
                        [0, 0, -2*zeta*wn]], dtype=np.float32)
        Bpos = np.diag([wn*wn, wn*wn, wn*wn]).astype(np.float32)

        pos = x[:3]
        vel = x[3:]

        pos_dot = Apos @ pos + Bpos @ u
        vel_dot = -1.5*vel + 3.0*(pos_dot - vel)

        pos = pos + DT * vel
        vel = vel + DT * vel_dot

        noise_pos = np.random.normal(0.0, [0.007, 0.007, 0.010]) * 0.0174533
        noise_vel = np.random.normal(0.0, [0.0012, 0.0012, 0.0012])

        pos = pos + noise_pos
        vel = vel + noise_vel

        x[:3] = pos
        x[3:] = vel
        self._state = x.astype(np.float32)

        roll_deg, pitch_deg = math.degrees(x[0]), math.degrees(x[1])
        ax = roll_deg * 0.0174 + np.random.normal(0.026, 0.006)
        ay = pitch_deg * 0.0174 + np.random.normal(0.005, 0.006)
        az = np.clip(0.970 - 0.5*(roll_deg**2 + pitch_deg**2)*0.0003, 0.85, 1.0)
        az += np.random.normal(0, 0.015)
        
        self._acc = np.array([ax, ay, az], dtype=np.float32)

        return dict(
            rpy=(x[0], x[1], x[2]),
            gyro=(x[3], x[4], x[5]),
            acc=tuple(self._acc.tolist())
        )

    def _imu_online_wait(self):
        imu = self.link.wait_first_imu()
        return imu

    def _imu_online_latest(self):
        imu = self.link.read_latest_imu(DT) or self.link.wait_first_imu()
        return imu

    def _obs_from_imu(self, imu):
        roll, pitch, yaw = imu["rpy"]
        gx, gy, gz = imu["gyro"]
        ax, ay, az = imu["acc"]
        return np.array([roll, pitch, yaw, gx, gy, gz, ax, ay, az, *self._theta_cmd], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._step_count = 0
        
        # FIX: Reset physics state properly
        self._position_x = 0.0
        self._last_position_x = 0.0
        self._velocity_x = 0.0  # CRITICAL: Reset velocity!
        self._cumulative_distance = 0.0
        self._state = np.zeros(6, dtype=np.float32)  # Reset orientation

        # FIX: Better initial pose - start with slight stance
        # Small negative angles for vertical = legs pushing down slightly
        self._theta_cmd[:6] = self.np_random.uniform(-0.1, 0.1, size=6).astype(np.float32)  # Horizontal
        self._theta_cmd[6:] = self.np_random.uniform(-0.2, -0.05, size=6).astype(np.float32)  # Vertical (down)

        if self.use_hardware:
            imu = self._imu_online_wait()
            self.link.send_joint_targets_rad12(self._theta_cmd.tolist())
        else:
            imu = self._imu_offline()

        return self._obs_from_imu(imu), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        theta = self._theta_cmd + action

        for i, (lo, hi) in enumerate(JOINT_LIMITS_12):
            theta[i] = float(np.clip(theta[i], lo, hi))
        self._theta_cmd = theta.astype(np.float32)

        if self.use_hardware:
            self.link.send_joint_targets_rad12(self._theta_cmd.tolist())

        imu = self._imu_online_latest() if self.use_hardware else self._imu_offline()
        obs = self._obs_from_imu(imu)

        # FIX: Completely redesigned reward function
        roll, pitch = obs[0], obs[1]
        qH = obs[9:15]
        qV = obs[15:21]
        
        # 1. FIX: Forward movement reward (DOMINANT)
        # Use velocity directly instead of tiny position deltas
        forward_distance = self._position_x - self._last_position_x
        self._cumulative_distance += forward_distance
        self._last_position_x = self._position_x
        
        # Reward both instantaneous velocity AND cumulative progress
        r_forward = 50.0 * max(self._velocity_x, 0.0)  # Reward forward velocity
        r_forward += 200.0 * forward_distance  # Big bonus for actual movement
        
        # FIX: Bonus for sustained forward progress
        if self._step_count > 0 and self._cumulative_distance > 0.01 * self._step_count:
            r_forward += 5.0  # Consistency bonus
        
        # 2. Stability (secondary)
        angle_mag = math.sqrt(roll**2 + pitch**2)
        r_upright = 2.0 * math.exp(-3.0 * angle_mag)  # Small but helps
        
        # 3. Energy efficiency
        r_smooth = -0.01 * float(np.sum(action**2))
        
        # 4. FIX: Encourage alternating gait pattern
        # Reward when left and right legs are doing opposite things
        left_h = np.mean(qH[[0, 2, 4]])
        right_h = np.mean(qH[[1, 3, 5]])
        r_gait = 3.0 * abs(left_h - right_h)  # Reward difference
        
        # 5. FIX: Encourage ground contact (negative vertical = pushing down)
        ground_pressure = -np.mean(qV)  # Negative is good
        r_ground = 2.0 * np.clip(ground_pressure, 0.0, 0.5)
        
        # Total reward
        reward = r_forward + r_upright + r_smooth + r_gait + r_ground

        terminated = (abs(math.degrees(roll)) > self.fall_deg) or (abs(math.degrees(pitch)) > self.fall_deg)
        truncated = (self._t >= self.steps_max)
        self._t += 1
        self._step_count += 1
        
        return obs, reward, terminated, truncated, {}

    def render(self): pass

    def close(self):
        if self.link is not None:
            self.link.close()
            self.link = None