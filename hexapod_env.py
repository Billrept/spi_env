# 12-dof env with two modes:
#   - use_hardware=False (default): no serial; mock IMU; train offline
#   - use_hardware=True:  serial IMU + remocon packets; run on robot
#
# TASK: Forward locomotion - move the spider robot forward using horizontal joints
# obs = IMU (9) + commanded angles (12) = 21-D
# actions = 12 deltas (radians): [H1..H6, V1..V6]

import gymnasium as gym
import numpy as np, math, time, random
from gymnasium import spaces

# ---- control & episode ----
CTRL_HZ = 50
DT = 1.0 / CTRL_HZ
EPISODE_SECONDS = 10.0
FALL_DEG = 60.0  # More lenient fall threshold for better learning

# per-step delta caps (can differ for H vs V if you want)
MAX_STEP_RAD_H = 0.05
MAX_STEP_RAD_V = 0.05

# real joint limits (safety clamp); center at 180° (2048 ticks) ± 45°
JOINT_LIMITS_12 = [(-0.785, 0.785)] * 12  # ±45° from center (135° to 225°)

class SPIBalance12Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, port=None, use_hardware: bool=False, seed: int | None=None,
                 episode_seconds: float=None, max_tilt_deg: float=None):
        super().__init__()
        self.use_hardware = use_hardware
        self.port = port
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        # Override episode duration and fall threshold if provided
        self.episode_seconds = episode_seconds if episode_seconds is not None else EPISODE_SECONDS
        self.fall_deg = max_tilt_deg if max_tilt_deg is not None else FALL_DEG
        
        self._state = np.zeros(6, dtype=np.float32)  # [roll, pitch, yaw, gx, gy, gz]
        self._acc   = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        # Track forward position for locomotion reward
        self._position_x = 0.0  # forward distance traveled
        self._last_position_x = 0.0
        # obs: [roll, pitch, yaw, gx, gy, gz, ax, ay, az, qH1..qH6, qV1..qV6] (21-D)
        high_imu = np.array([math.pi]*3 + [50.0]*3 + [5.0]*3, dtype=np.float32)
        high_q   = np.array([2.5]*12, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.concatenate([high_imu, high_q]),
            high=np.concatenate([high_imu, high_q]),
            dtype=np.float32
        )

        # actions: 12 deltas (H first, then V)
        low  = np.array([-MAX_STEP_RAD_H]*6 + [-MAX_STEP_RAD_V]*6, dtype=np.float32)
        high = np.array([+MAX_STEP_RAD_H]*6 + [+MAX_STEP_RAD_V]*6, dtype=np.float32)
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        # commanded joint angles (what we send/store)
        self._theta_cmd = np.zeros(12, dtype=np.float32)
        self._t = 0
        self.steps_max = int(self.episode_seconds * CTRL_HZ)

        # online link only when needed
        self.link = None
        if self.use_hardware:
            from textimu_link import Remocon12Link
            self.link = Remocon12Link(port=self.port)

        self._imu_bias = np.zeros(9, dtype=np.float32)  # [r,p,y,gx,gy,gz,ax,ay,az]
        self._imu_sigma = np.array([0.02,0.02,0.03,  0.5,0.5,0.5,  0.01,0.01,0.01], dtype=np.float32)

    # ---------- IMU providers ----------
    def _imu_offline(self):
        """
        IMPROVED hexapod dynamics with:
        1. More consistent acceleration (less velocity decay)
        2. Better force accumulation model
        3. Minimum forward bias to encourage exploration
        """
        x = self._state.copy()
        qH = self._theta_cmd[:6]    # horizontal
        qV = self._theta_cmd[6:]    # vertical

        # ===== IMPROVED FORWARD LOCOMOTION MECHANICS =====
        # Key changes:
        # 1. Reduced drag to maintain velocity longer
        # 2. Added small forward bias to break symmetry
        # 3. More linear force-to-velocity coupling
        
        forward_force = 0.0
        
        # Calculate push force from each leg
        for i in range(6):
            is_left_leg = (i % 2 == 0)
            
            # Horizontal push direction (same as before)
            if is_left_leg:
                push_direction = -np.sin(qH[i])
            else:
                push_direction = +np.sin(qH[i])
            
            # IMPROVED: More responsive ground pressure model
            # Vertical angle: negative = extended down = strong push
            ground_pressure = np.clip(1.0 - 1.5 * qV[i], 0.0, 2.0)
            
            # Leg force contribution
            leg_force = push_direction * ground_pressure
            forward_force += leg_force
        
        # Normalize by number of legs and add SMALL forward bias
        # This breaks symmetry and encourages forward exploration
        forward_force = (forward_force / 6.0) * 3.0
        forward_force += 0.05  # Small constant forward bias
        
        # ===== IMPROVED DYNAMICS =====
        MASS = 1.5              # kg
        FRICTION = 0.88         # Slightly higher friction
        DRAG = 0.96             # MUCH less drag (was 0.92)
        MAX_SPEED = 0.8         # m/s
        MIN_SPEED = -0.2        # Allow small backward motion
        
        # Calculate acceleration
        forward_accel = (forward_force * FRICTION) / MASS
        
        # Initialize velocity if needed
        if not hasattr(self, '_velocity_x'):
            self._velocity_x = 0.0
        
        # IMPROVED: Velocity update with less aggressive decay
        self._velocity_x = self._velocity_x * DRAG + forward_accel * DT
        
        # Clamp velocity
        self._velocity_x = np.clip(self._velocity_x, MIN_SPEED, MAX_SPEED)
        
        # Update position
        self._position_x += self._velocity_x * DT
        
        # Track acceleration for reward shaping
        if not hasattr(self, '_prev_velocity'):
            self._prev_velocity = 0.0
        self._acceleration_x = (self._velocity_x - self._prev_velocity) / DT
        self._prev_velocity = self._velocity_x

        # ===== BALANCE DYNAMICS (unchanged) =====
        left_legs_v  = qV[[0, 2, 4]]
        right_legs_v = qV[[1, 3, 5]]
        front_legs_v = qV[[0, 1]]
        rear_legs_v  = qV[[4, 5]]

        r_cmd = 0.5 * (np.mean(right_legs_v) - np.mean(left_legs_v))
        p_cmd = 0.5 * (np.mean(front_legs_v) - np.mean(rear_legs_v))
        
        left_legs_h = qH[[0, 2, 4]]
        right_legs_h = qH[[1, 3, 5]]
        y_cmd = 0.3 * (np.mean(left_legs_h) - np.mean(right_legs_h))

        u = np.array([r_cmd, p_cmd, y_cmd], dtype=np.float32)

        # First-order dynamics
        zeta = 0.7
        wn   = 6.0
        Apos = np.array([[-2*zeta*wn, 0, 0],
                        [0, -2*zeta*wn, 0],
                        [0, 0, -2*zeta*wn]], dtype=np.float32)
        Bpos = np.diag([wn*wn, wn*wn, wn*wn]).astype(np.float32)

        pos = x[:3]
        vel = x[3:]

        pos_dot = Apos @ pos + Bpos @ u
        vel_dot = -1.5*vel + 3.0*(pos_dot - vel)

        # Integrate
        pos = pos + DT * vel
        vel = vel + DT * vel_dot

        # Add realistic noise
        noise_pos = np.random.normal(0.0, [0.007, 0.007, 0.010])
        noise_pos = noise_pos * 0.0174533
        noise_vel = np.random.normal(0.0, [0.0012, 0.0012, 0.0012])

        pos = pos + noise_pos
        vel = vel + noise_vel

        # Save state
        x[:3] = pos
        x[3:] = vel
        self._state = x.astype(np.float32)

        # Acc vector (unchanged)
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

    # ---------- obs pack ----------
    def _obs_from_imu(self, imu):
        roll, pitch, yaw = imu["rpy"]
        gx, gy, gz = imu["gyro"]
        ax, ay, az = imu["acc"]
        return np.array([roll, pitch, yaw, gx, gy, gz, ax, ay, az, *self._theta_cmd], dtype=np.float32)

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        
        # Reset position tracking for locomotion
        self._position_x = 0.0
        self._last_position_x = 0.0

        # initialize commanded pose with small randomization for better exploration
        # Start near neutral with slight variations
        self._theta_cmd[:] = self.np_random.uniform(-0.05, 0.05, size=12).astype(np.float32)

        if self.use_hardware:
            imu = self._imu_online_wait()
            # send current command (no jump)
            self.link.send_joint_targets_rad12(self._theta_cmd.tolist())
        else:
            imu = self._imu_offline()

        return self._obs_from_imu(imu), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        theta  = self._theta_cmd + action

        # clamp to hard limits
        for i, (lo, hi) in enumerate(JOINT_LIMITS_12):
            theta[i] = float(np.clip(theta[i], lo, hi))
        self._theta_cmd = theta.astype(np.float32)

        # send if online
        if self.use_hardware:
            self.link.send_joint_targets_rad12(self._theta_cmd.tolist())

        # get IMU
        imu = self._imu_online_latest() if self.use_hardware else self._imu_offline()
        obs = self._obs_from_imu(imu)

        # ===== REWARD FUNCTION: Encourage Forward Locomotion with Wave Gait =====
        roll, pitch = obs[0], obs[1]
        qH = obs[9:15]   # horizontal joint angles [FL, FR, ML, MR, RL, RR]
        qV = obs[15:21]  # vertical joint angles
        
        # 1. Forward progress reward (PRIMARY OBJECTIVE - DOMINANT REWARD)
        #    Must be MUCH larger than other rewards to be effective
        #    Scale by 1000x to make forward movement the main objective
        forward_distance = self._position_x - self._last_position_x
        r_forward = 1000.0 * forward_distance  # MASSIVELY increased from 15x
        self._last_position_x = self._position_x
        
        # 2. Stability reward - stay upright while moving (normalized to 0-1 range)
        #    Exponential penalty for tilting
        angle_mag = math.sqrt(roll**2 + pitch**2)
        r_upright = 0.1 * math.exp(-2.5 * angle_mag)  # Reduced from 1.0 to 0.1
        
        # 3. Energy efficiency - small penalty for large actions
        r_smooth = -0.001 * float(np.sum(action**2))  # Reduced from -0.01
        
        # 4. Gait coordination reward - encourage wave pattern
        #    High variance means legs moving differently (coordinated gait)
        qH_variance = float(np.var(qH))
        r_gait = 0.05 * min(qH_variance, 0.4)  # Reduced from 0.8
        
        # 4b. Bonus for opposite leg pairs (left vs right coordination)
        #     FL vs FR, ML vs MR, RL vs RR should have opposite signs for balanced push
        pair_opposites = -(qH[0] * qH[1]) - (qH[2] * qH[3]) - (qH[4] * qH[5])
        r_pair_coordination = 0.02 * np.clip(pair_opposites, -0.5, 0.5)  # Reduced from 0.3
        
        # 5. Vertical movement penalty - minimize excessive vertical joint usage
        r_vertical_penalty = -0.001 * float(np.sum(qV**2))  # Reduced from -0.015
        
        # Total reward (forward movement heavily dominates)
        reward = r_forward + r_upright + r_smooth + r_gait + r_pair_coordination + r_vertical_penalty

        terminated = (abs(math.degrees(roll)) > self.fall_deg) or (abs(math.degrees(pitch)) > self.fall_deg)
        truncated  = (self._t >= self.steps_max)
        self._t += 1
        return obs, reward, terminated, truncated, {}

    def render(self): pass

    def close(self):
        if self.link is not None:
            self.link.close()
            self.link = None
