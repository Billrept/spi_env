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
        self._total_reward = 0.0
        self._movement_scale = 0.0005  # Reduced for more realistic movement
        
        # Initialize observation space
        obs_size = 21  # 9 IMU values + 12 joint angles
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        low  = np.array([-MAX_STEP_RAD_H]*6 + [-MAX_STEP_RAD_V]*6, dtype=np.float32)
        high = np.array([+MAX_STEP_RAD_H]*6 + [+MAX_STEP_RAD_V]*6, dtype=np.float32)
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        # Initialize state variables
        self._theta_cmd = np.zeros(12, dtype=np.float32)
        self._last_imu = None  # FIX: Initialize with None

        self._t = 0
        self.steps_max = int(self.episode_seconds * CTRL_HZ)

        self.link = None
        if self.use_hardware:
            from textimu_link import Remocon12Link
            self.link = Remocon12Link(port=self.port)

        # Set IMU function based on hardware flag
        self._imu_fn = self._imu_offline if not use_hardware else self._imu_online
        
        # Initialize position tracking
        self._last_position_x = 0.0
        self._position_x = 0.0
        self._cumulative_distance = 0.0
        self._total_reward = 0.0
        self._movement_scale = 0.0005
        
    def _imu_offline(self):
        """Mock IMU data for offline training"""
        return {
            'ax': 0.0,     # Acceleration in m/s^2
            'ay': 0.0,
            'az': -9.81,   # Gravity
            'gx': 0.0,     # Angular velocity in rad/s
            'gy': 0.0,
            'gz': 0.0,
            'roll': 0.0,   # Orientation in radians
            'pitch': 0.0,
            'yaw': 0.0
        }

    def _get_obs(self, imu):
        """Convert IMU and joint data into observation vector"""
        # IMU data (9 values)
        imu_data = np.array([
            imu['ax'], imu['ay'], imu['az'],    # Accelerometer
            imu['gx'], imu['gy'], imu['gz'],    # Gyroscope
            imu['roll'], imu['pitch'], imu['yaw']  # Orientation
        ], dtype=np.float32)
        
        # Combine IMU and joint angles
        obs = np.concatenate([
            imu_data,           # 9 IMU values
            self._theta_cmd     # 12 joint angles
        ])
        
        return obs.astype(np.float32)

    def _imu_offline(self):
        """ULTRA-SIMPLIFIED physics to debug position tracking"""
        x = self._state.copy()
        qH = self._theta_cmd[:6]
        qV = self._theta_cmd[6:]

        # ===== FORCE CALCULATION =====
        # Physical model: Legs sweeping backward push the body forward
        # Left legs: negative angle = backward sweep = forward push
        # Right legs: positive angle = backward sweep = forward push
        
        left_legs = qH[[0, 2, 4]]   # FL, ML, RL
        right_legs = qH[[1, 3, 5]]  # FR, MR, RR
        
        # Convert joint angles to forward push force
        # Negative left angles = backward sweep = positive forward force
        # Positive right angles = backward sweep = positive forward force
        left_push = -np.sum(left_legs)    # Negate because negative input should give positive output
        right_push = np.sum(right_legs)   # Positive input gives positive output
        total_push = (left_push + right_push) / 6.0  # Average per leg
        
        # Ground contact from vertical joints (negative = pushing down = good)
        ground_pressure = np.mean(np.clip(-qV, 0.0, 0.785))
        
        # CRITICAL: MUCH stronger force multiplier for visible movement
        FORCE_MULTIPLIER = 300.0  # Need big movements for RL!
        forward_force = total_push * (0.5 + ground_pressure) * FORCE_MULTIPLIER
        
        # ===== PHYSICS INTEGRATION =====
        MASS = 0.8   # Lighter for more responsive movement
        FRICTION = 0.95
        DRAG = 0.75  # More drag to stabilize
        
        # F = ma
        accel = (forward_force * FRICTION) / MASS
        
        # Update velocity
        self._velocity_x = float(self._velocity_x * DRAG + accel * DT)
        self._velocity_x = np.clip(self._velocity_x, -0.5, 1.5)
        
        # Update position
        delta_x = self._velocity_x * DT
        self._position_x = float(self._position_x + delta_x)
        
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

    def reset(self, *, seed=None, options=None):
        
        super().reset(seed=seed)
        self._t = 0
        self._step_count = 0
        
        # FIX: Reset physics state properly
        self._position_x = 0.0
        self._last_position_x = 0.0
        self._velocity_x = 0.0  # CRITICAL: Reset velocity!
        self._cumulative_distance = 0.0
        self._total_reward = 0.0
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

        return self._get_obs(imu), {}

    def step(self, action):
        self._step_count += 1
        self._last_position_x = self._position_x
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        theta = self._theta_cmd + action

        for i, (lo, hi) in enumerate(JOINT_LIMITS_12):
            theta[i] = float(np.clip(theta[i], lo, hi))
        self._theta_cmd = theta.astype(np.float32)

        if self.use_hardware:
            self.link.send_joint_targets_rad12(self._theta_cmd.tolist())

        imu = self._imu_online_latest() if self.use_hardware else self._imu_offline()
        obs = self._get_obs(imu)

        # Calculate movement based on action direction
        action_mean = float(np.mean(action))
        dx = action_mean * self._movement_scale
        
        # Update position and cumulative distance
        self._position_x += dx
        self._cumulative_distance += abs(dx)
        
        # Calculate reward with better scaling
        # Larger reward for forward motion, small penalty for backward
        if dx > 0:
            reward = dx * 1000.0  # Scale up forward reward
        else:
            reward = dx * 100.0   # Smaller penalty for backward
            
        self._total_reward += reward
        
        # Get observations and check termination
        imu = self._imu_fn()
        obs = self._get_obs(imu)
        terminated = self._check_terminated(imu)
        truncated = (self._step_count >= self._max_steps)
        
        info = {
            'position_x': self._position_x,
            'cumulative_distance': self._cumulative_distance,
            'dx': dx,
            'total_reward': self._total_reward,
            'action_mean': action_mean,
            'step_count': self._step_count
        }
        
        return obs, reward, terminated, truncated, info

    def render(self): pass

    def close(self):
        if self.link is not None:
            self.link.close()
            self.link = None