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

# real joint limits (safety clamp); match remocon12_link
JOINT_LIMITS_12 = [(-2.094, 2.094)] * 12  # ±120°

class SPIBalance12Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, port=None, use_hardware: bool=False, seed: int | None=None):
        super().__init__()
        self.use_hardware = use_hardware
        self.port = port
        self.np_random, _ = gym.utils.seeding.np_random(seed)
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
        self.steps_max = int(EPISODE_SECONDS * CTRL_HZ)

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
        Toy dynamics: roll/pitch/yaw respond to commanded joint angles.
        State x = [r, p, y, gx, gy, gz]; u = 12 commanded angles (H1..H6, V1..V6).
        """
        x = self._state.copy()
        qH = self._theta_cmd[:6]    # horizontal
        qV = self._theta_cmd[6:]    # vertical

        # Aggregate "inputs" from joints
        # Left legs = 0,2,4 ; Right legs = 1,3,5
        r_cmd = 0.6 * (np.mean(qV[[1,3,5]]) - np.mean(qV[[0,2,4]]))    # right - left
        p_cmd = 0.6 * (np.mean(qV[[0,1]])   - np.mean(qV[[4,5]]))      # front - back (00/11 vs 44/55; reindex as needed)
        y_cmd = 0.5 * (np.mean(qH[:3]) - np.mean(qH[3:]))              # H front group - back group

        # Forward velocity from horizontal joint movement (gait pattern)
        # Coordinated horizontal movement creates forward motion
        qH_variance = np.var(qH)  # variety in horizontal positions
        qH_mean_abs = np.mean(np.abs(qH))  # how much horizontal joints are active
        forward_vel = 0.3 * qH_variance + 0.2 * qH_mean_abs  # forward speed proxy
        
        u = np.array([r_cmd, p_cmd, y_cmd], dtype=np.float32)

        # First-order dynamics parameters
        zeta = 0.7          # damping
        wn   = 6.0          # nat. freq (rad/s)
        Apos = np.array([[-2*zeta*wn, 0,            0],
                        [0,          -2*zeta*wn,   0],
                        [0,           0,          -2*zeta*wn]], dtype=np.float32)
        Bpos = np.diag([wn*wn, wn*wn, wn*wn]).astype(np.float32)

        # Continuous-time: [r,p,y]_dot = Apos*[r,p,y] + Bpos*u
        pos = x[:3]
        vel = x[3:]

        pos_dot = Apos @ pos + Bpos @ u
        vel_dot = -1.5*vel + 3.0*(pos_dot - vel)   # crude vel relaxation toward pos_dot

        # Integrate (Euler)
        pos = pos + DT * vel
        vel = vel + DT * vel_dot

        # Add tiny noise
        noise_pos = np.random.normal(0.0, [0.003,0.003,0.005])
        noise_vel = np.random.normal(0.0, [0.05,0.05,0.05])

        pos = pos + noise_pos
        vel = vel + noise_vel

        # Save state
        x[:3] = pos
        x[3:] = vel
        self._state = x.astype(np.float32)
        
        # Update forward position based on horizontal joint movement
        self._position_x += forward_vel * DT

        # Acc vector ~ gravity in body frame (approx small angles)
        ax = 0.0 + np.random.normal(0, 0.005)
        ay = 0.0 + np.random.normal(0, 0.005)
        az = 1.0 + np.random.normal(0, 0.005)
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

        # reward: forward movement + upright + smooth + gait coordination
        roll, pitch = obs[0], obs[1]
        gyro_x, gyro_y, gyro_z = obs[3], obs[4], obs[5]
        qH = obs[9:15]  # horizontal joint angles from observation
        
        # 1. Forward velocity reward (primary objective)
        forward_distance = self._position_x - self._last_position_x
        r_forward = 10.0 * forward_distance  # high weight on forward progress
        self._last_position_x = self._position_x
        
        # 2. Upright reward - don't fall over while moving
        angle_mag = math.sqrt(roll**2 + pitch**2)
        r_upright = 0.5 * math.exp(-2.0 * angle_mag)
        
        # 3. Smooth actions - penalize jerky movements
        r_smooth = -0.005 * float(np.sum(action**2))
        
        # 4. Gait coordination bonus - reward variance in horizontal joints (walking pattern)
        qH_variance = float(np.var(qH))
        r_gait = 0.5 * min(qH_variance, 0.5)  # cap to prevent excessive movement
        
        # 5. Energy efficiency - penalize excessive vertical movement
        qV = obs[15:21]  # vertical joint angles
        r_vertical_penalty = -0.02 * float(np.sum(qV**2))
        
        reward = r_forward + r_upright + r_smooth + r_gait + r_vertical_penalty

        terminated = (abs(math.degrees(roll)) > FALL_DEG) or (abs(math.degrees(pitch)) > FALL_DEG)
        truncated  = (self._t >= self.steps_max)
        self._t += 1
        return obs, reward, terminated, truncated, {}

    def render(self): pass

    def close(self):
        if self.link is not None:
            self.link.close()
            self.link = None
