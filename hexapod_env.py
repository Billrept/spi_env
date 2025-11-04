# 12-dof env with two modes:
#   - use_hardware=False (default): no serial; mock IMU; train offline
#   - use_hardware=True:  serial IMU + remocon packets; run on robot
#
# TASK: Forward locomotion - move the spider robot forward using horizontal joints
# obs = IMU (9) + commanded angles (12) = 21-D OR 45-D (with feedback)
# actions = 12 deltas (radians): [H1..H6, V1..V6]

import gymnasium as gym
import numpy as np, math, time, random
from gymnasium import spaces
from textimu_link import CH_MAP_12, INVERT_12, TICKS2RAD, JOINT_LIMITS_12

# ---- control & episode ----
CTRL_HZ = 50
DT = 1.0 / CTRL_HZ
EPISODE_SECONDS = 10.0
FALL_DEG = 60.0  # More lenient fall threshold for better learning

# per-step delta caps (can differ for H vs V if you want)
MAX_STEP_RAD_H = 0.05
MAX_STEP_RAD_V = 0.05

# Joint limits imported from textimu_link (front legs have extended range)
# JOINT_LIMITS_12 is defined in textimu_link.py

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
        self._theta_actual = np.zeros(12, dtype=np.float32)  # Actual servo positions (hardware feedback)
        
        # obs: REDUCED to only front legs + vertical (hide other horizontal from policy!)
        # [IMU(9), FLH, FRH, ALL_VERTICAL(6), FLH_actual, FRH_actual, FLH_error, FRH_error]
        # Total: 9 + 2 + 6 + 2 + 2 = 21-D 
        high_imu = np.array([math.pi]*3 + [50.0]*3 + [5.0]*3, dtype=np.float32)
        high_front_H = np.array([1.2]*2, dtype=np.float32)  # FLH, FRH: ±69° extended range
        high_all_V = np.array([0.785]*6, dtype=np.float32)     # All 6 vertical: ±45°
        high_front_error = np.array([2.4]*2, dtype=np.float32)  # FLH, FRH tracking error (2x range)
        self.observation_space = spaces.Box(
            low=-np.concatenate([high_imu, high_front_H, high_all_V, high_front_H, high_front_error]),
            high=np.concatenate([high_imu, high_front_H, high_all_V, high_front_H, high_front_error]),
            dtype=np.float32
        )

        # actions: 8 deltas (only FLH, FRH + 6 vertical joints)
        # [FLH, FRH, FLV, FRV, MLV, MRV, RLV, RRV]
        low  = np.array([-MAX_STEP_RAD_H]*2 + [-MAX_STEP_RAD_V]*6, dtype=np.float32)
        high = np.array([+MAX_STEP_RAD_H]*2 + [+MAX_STEP_RAD_V]*6, dtype=np.float32)
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        # commanded joint angles (what we send/store)
        self._theta_cmd = np.zeros(12, dtype=np.float32)
        self._theta_cmd_prev = np.zeros(12, dtype=np.float32)  # For oscillation reward
        self._t = 0
        self.steps_max = int(self.episode_seconds * CTRL_HZ)
        self._debug_counter = 0

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
        Improved hexapod dynamics with ground contact simulation.
        State x = [r, p, y, gx, gy, gz]; u = 12 commanded angles (H1..H6, V1..V6).
        
        FRONT-LEG CRAWL MODE: Only FLH and FRH contribute to physics!
        Middle and rear horizontal joints are ZEROED in simulation.
        """
        x = self._state.copy()
        qH = self._theta_cmd[:6].copy()    # horizontal [FLH, FRH, MLH, MRH, RLH, RRH]
        qV = self._theta_cmd[6:]    # vertical

        qH[2] = 0.0  # MLH = 0
        qH[3] = 0.0  # MRH = 0
        qH[4] = 0.0  # RLH = 0
        qH[5] = 0.0  # RRH = 0

        # ===== HEXAPOD KINEMATICS =====
        # Simplified leg model: 2-DOF (horizontal sweep + vertical lift)
        LEG_LENGTH = 0.15  # meters (adjust to your robot)
        
        # FIXED: Assume spider always walks on ground (realistic for hexapod)
        # Vertical joints control leg extension/compression (how hard legs push)
        # Horizontal joints control sweep direction (which way to push)
        
        # ===== FORWARD LOCOMOTION =====
        # All legs assumed on ground (hexapod is stable platform)
        # Push force = horizontal sweep direction × vertical pressure

        # CRITICAL: Left and right legs push in OPPOSITE directions!
        # Leg mapping: 0=FL, 1=FR, 2=ML, 3=MR, 4=RL, 5=RR
        # Left legs (0,2,4): Negative qH = sweep outward/back = push forward
        # Right legs (1,3,5): POSITIVE qH = sweep inward/back = push forward

        # ===== FORWARD LOCOMOTION MECHANICS =====
        # Calculate net forward force from all 6 legs
        # Key insight: Coordinated pairs create balanced forward push

        forward_force = 0.0
        for i in range(6):
            # Determine if this is a left or right leg
            is_left_leg = (i % 2 == 0)  # 0,2,4 are left; 1,3,5 are right
            
            # Horizontal angle determines push direction
            # Wave gait: pairs move together (FL+FR, ML+MR, RL+RR)
            # Left legs: negative qH = backward sweep → forward push
            # Right legs: positive qH = backward sweep → forward push (mirrored)
            if is_left_leg:
                push_direction = -np.sin(qH[i])  # Left: negative angle = forward
            else:
                push_direction = +np.sin(qH[i])  # Right: positive angle = forward

            # Vertical angle determines push strength (ground contact pressure)
            # Negative qV = leg extended down → strong push
            # Positive qV = leg lifted up → weak/no push (recovery phase)
            # Enhanced pressure model for more responsive locomotion
            ground_pressure = np.clip(0.8 - 1.2 * qV[i], 0.0, 1.5)

            # Combined force from this leg (proportional to angle × pressure)
            leg_force = push_direction * ground_pressure
            forward_force += leg_force
        
        # Scale by number of legs (normalize) with STRONG boost for active pushing
        # Multiply by 3.0 to make movements much more effective (increased from 1.5)
        forward_force = (forward_force / 6.0) * 3.0

        # Body dynamics - highly responsive for clear RL feedback
        MASS = 1.0
        FRICTION = 0.85      # high grip for effective pushing (was 0.75)
        DRAG = 0.92          # more momentum preserved (was 0.88)
        MAX_SPEED = 0.84      # m/s (increased from 0.6 for faster locomotion)
        
        # Calculate acceleration (F = ma)
        forward_accel = (forward_force * FRICTION) / MASS
        
        # Initialize velocity tracking if needed
        if not hasattr(self, '_velocity_x'):
            self._velocity_x = 0.0

        # Update velocity: v_new = v_old * drag + accel * dt
        self._velocity_x = self._velocity_x * DRAG + forward_accel * DT

        # Clamp to realistic hexapod speed range
        self._velocity_x = np.clip(self._velocity_x, -0.3, MAX_SPEED)
        
        # Update position
        forward_vel = self._velocity_x

        # ===== BALANCE DYNAMICS =====
        # Leg mapping: 0=FL, 1=FR, 2=ML, 3=MR, 4=RL, 5=RR
        # Left legs = 0(FL), 2(ML), 4(RL)
        # Right legs = 1(FR), 3(MR), 5(RR)
        left_legs_v  = qV[[0, 2, 4]]   # FL, ML, RL vertical
        right_legs_v = qV[[1, 3, 5]]   # FR, MR, RR vertical
        front_legs_v = qV[[0, 1]]      # FL, FR vertical
        rear_legs_v  = qV[[4, 5]]      # RL, RR vertical

        # Center of mass shift due to leg asymmetry
        r_cmd = 0.5 * (np.mean(right_legs_v) - np.mean(left_legs_v))   # right - left
        p_cmd = 0.5 * (np.mean(front_legs_v) - np.mean(rear_legs_v))   # front - back

        # Yaw from horizontal leg asymmetry
        left_legs_h = qH[[0, 2, 4]]
        right_legs_h = qH[[1, 3, 5]]
        y_cmd = 0.3 * (np.mean(left_legs_h) - np.mean(right_legs_h))

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

        # Add realistic sensor noise (CALIBRATED TO REAL ROBOT)
        # Real standing still gyro: -0.05 to +0.14 dps ≈ -0.001 to +0.0024 rad/s
        # Convert to radians: ±0.07 dps ≈ ±0.0012 rad/s typical variation
        noise_pos = np.random.normal(0.0, [0.007, 0.007, 0.010])  # degrees, matches real 0.01° noise
        noise_pos = noise_pos * 0.0174533  # convert to radians
        
        # Real gyro noise when standing: ±0.07 dps = ±0.0012 rad/s
        noise_vel = np.random.normal(0.0, [0.0012, 0.0012, 0.0012])  # rad/s

        pos = pos + noise_pos
        vel = vel + noise_vel

        # Save state
        x[:3] = pos
        x[3:] = vel
        self._state = x.astype(np.float32)
        
        # Update forward position based on horizontal joint movement
        self._position_x += forward_vel * DT

        # Acc vector ~ gravity in body frame (CALIBRATED TO REAL ROBOT)
        # Real standing still: R/P/Y ≈ 0.39°, -1.54°, 14.24° | ACC ≈ 0.026, 0.005, 0.970
        # This shows natural tilt and sensor noise characteristics
        roll_deg, pitch_deg = math.degrees(x[0]), math.degrees(x[1])
        
        # Gravity projection based on body orientation (small angle approximation)
        # Real robot shows: ax ≈ 0.026±0.006, ay ≈ 0.005±0.006, az ≈ 0.970±0.015
        ax = roll_deg * 0.0174 + np.random.normal(0.026, 0.006)      # g, with realistic offset
        ay = pitch_deg * 0.0174 + np.random.normal(0.005, 0.006)     # g
        az = np.clip(0.970 - 0.5*(roll_deg**2 + pitch_deg**2)*0.0003, 0.85, 1.0)  # decreases with tilt
        az += np.random.normal(0, 0.015)  # Real sensor noise
        
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
        
        # Extract actual servo positions if available (for hardware feedback)
        if "servo_positions" in imu and self.use_hardware:
            servo_dict = imu["servo_positions"]
            # Convert motor IDs to joint angles in our joint order
            theta_actual = np.zeros(12, dtype=np.float32)
            for joint_idx in range(12):
                motor_id = CH_MAP_12[joint_idx]
                if motor_id in servo_dict:
                    ticks = servo_dict[motor_id]
                    # Convert ticks to radians (center=2048, ±45° = ±651 ticks)
                    angle_rad = (ticks - 2048) * TICKS2RAD
                    # Apply inversion if needed
                    if INVERT_12[joint_idx]:
                        angle_rad = -angle_rad
                    theta_actual[joint_idx] = angle_rad
                else:
                    # Fallback to commanded if feedback not available
                    theta_actual[joint_idx] = self._theta_cmd[joint_idx]
            
            # Store for tracking error calculation (but IGNORE ML/MR/RL/RR actual values!)
            self._theta_actual = theta_actual
            
            # Extract horizontal and vertical components
            qH = self._theta_cmd[0::2]  # Horizontal: [FLH, FRH, MLH, MRH, RLH, RRH]
            qV = self._theta_cmd[1::2]  # Vertical: [FLV, FRV, MLV, MRV, RLV, RRV]
            
            # FRONT-LEG CRAWL: Only use actual feedback for FLH and FRH
            # For tracking error calculation, use commanded values for all other joints
            qH_actual = theta_actual[0::2]  # [FLH_actual, FRH_actual, MLH_actual, MRH_actual, RLH_actual, RRH_actual]
            qV_actual = theta_actual[1::2]  # All 6 vertical actual values
            
            # Observation (21-D): [IMU(9), FLH, FRH, V(6), FLH_actual, FRH_actual, FLH_error, FRH_error]
            # Only expose front horizontal joints (FLH, FRH) at indices 0,1
            tracking_error_front = qH[:2] - qH_actual[:2]  # Only FLH and FRH tracking error
            return np.array([roll, pitch, yaw, gx, gy, gz, ax, ay, az,
                           qH[0], qH[1],           # FLH, FRH commanded
                           *qV,                    # All 6 vertical joints
                           qH_actual[0], qH_actual[1],  # FLH, FRH actual
                           *tracking_error_front], dtype=np.float32)  # FLH, FRH tracking error
        else:
            # Simulation mode or no servo feedback: assume perfect tracking
            qH = self._theta_cmd[0::2]  # Horizontal: [FLH, FRH, MLH, MRH, RLH, RRH]
            qV = self._theta_cmd[1::2]  # Vertical: [FLV, FRV, MLV, MRV, RLV, RRV]
            
            # Observation (21-D): [IMU(9), FLH, FRH, V(6), FLH_actual, FRH_actual, FLH_error, FRH_error]
            # Zero tracking error in simulation
            zero_error_front = np.zeros(2, dtype=np.float32)
            return np.array([roll, pitch, yaw, gx, gy, gz, ax, ay, az,
                             qH[0], qH[1],           # FLH, FRH commanded
                             *qV,                    # All 6 vertical joints
                             qH[0], qH[1],           # FLH, FRH actual (same as commanded)
                             *zero_error_front], dtype=np.float32)  # FLH, FRH tracking error (zero)

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
        self._theta_cmd_prev[:] = self._theta_cmd.copy()  # Initialize prev state

        if self.use_hardware:
            imu = self._imu_online_wait()
            # send current command (no jump)
            self.link.send_joint_targets_rad12(self._theta_cmd.tolist())
        else:
            imu = self._imu_offline()

        return self._obs_from_imu(imu), {}

    def step(self, action):
        # action is 8-D: [FLH, FRH, FLV, FRV, MLV, MRV, RLV, RRV]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Map 8-D actions to 12-D joint deltas
        # Joint order: [FLH, FRH, MLH, MRH, RLH, RRH, FLV, FRV, MLV, MRV, RLV, RRV]
        action_12 = np.zeros(12, dtype=np.float32)
        action_12[0] = action[0]  # FLH
        action_12[1] = action[1]  # FRH
        # action_12[2:6] remain 0.0 (MLH, MRH, RLH, RRH frozen)
        action_12[6:12] = action[2:8]  # All 6 vertical joints
        
        theta = self._theta_cmd + action_12

        # clamp to hard limits
        for i, (lo, hi) in enumerate(JOINT_LIMITS_12):
            theta[i] = float(np.clip(theta[i], lo, hi))
        self._theta_cmd = theta.astype(np.float32)

        # send if online
        if self.use_hardware:
            # Enable debug every 50 steps
            debug = (self._debug_counter % 50 == 0)
            self._debug_counter += 1
            self.link.send_joint_targets_rad12(self._theta_cmd.tolist(), debug=debug)

        # get IMU
        imu = self._imu_online_latest() if self.use_hardware else self._imu_offline()
        obs = self._obs_from_imu(imu)

        # ===== REWARD FUNCTION: Front-Leg Crawl (SIMPLEST!) =====
        # Goal: Learn the ABSOLUTE SIMPLEST locomotion:
        #   Only FLH and FRH move (front two legs)
        #   All other legs stay at neutral (act as support)
        # Pattern: Front legs pull forward together → body crawls forward
        
        roll, pitch = obs[0], obs[1]
        # Observation now only exposes FLH, FRH in horizontal
        # obs[9:11] = [FLH, FRH] commanded
        # obs[11:17] = all 6 vertical joints
        FLH = obs[9]   # Front-Left Horizontal
        FRH = obs[10]  # Front-Right Horizontal
        qV = obs[11:17]  # vertical joint angles [FLV, FRV, MLV, MRV, RLV, RRV]
        
        # 1. Forward progress reward (PRIMARY OBJECTIVE - DOMINANT REWARD)
        forward_distance = self._position_x - self._last_position_x
        r_forward = 1000.0 * forward_distance
        self._last_position_x = self._position_x
        
        # 2. Stability reward - stay upright (critical since only 2 legs moving)
        angle_mag = math.sqrt(roll**2 + pitch**2)
        r_upright = 2.0 * math.exp(-2.5 * angle_mag)  # Increased to 2.0 (very important!)
        
        # 3. Front leg coordination - encourage FLH and FRH to move together
        #    Both front legs should have similar angles (synchronized crawl)
        front_diff = abs(FLH - FRH)
        r_front_sync = 1.0 * math.exp(-10.0 * front_diff)  # Strong reward when similar
        
        # 4. Front leg oscillation - encourage rhythmic back-and-forth
        #    Only track front legs (indices 0, 1 in full theta_cmd)
        FL_change = abs(self._theta_cmd[0] - self._theta_cmd_prev[0])
        FR_change = abs(self._theta_cmd[1] - self._theta_cmd_prev[1])
        mean_front_motion = (FL_change + FR_change) / 2.0
        r_front_oscillation = 3.0 * np.clip(mean_front_motion, 0, 0.3)  # Strong reward for movement
        
        
        # 6. Keep ALL vertical legs down - no lifting needed for crawl
        mean_qV_abs = float(np.mean(np.abs(qV)))
        r_vertical = -0.2 * mean_qV_abs  # Stronger penalty than before
        
        # 6. Energy efficiency - penalize large actions
        r_smooth = -0.001 * float(np.sum(action**2))
        
        # 7. STRONG saturation penalty - heavily penalize ANY joints at limits
        # Check against actual joint limits (accounts for asymmetric rear leg limits)
        at_limits_all = 0
        for i in range(12):
            lo, hi = JOINT_LIMITS_12[i]
            # Consider "at limit" if within 5% of range from either limit
            range_size = hi - lo
            threshold = 0.05 * range_size
            if self._theta_cmd[i] < (lo + threshold) or self._theta_cmd[i] > (hi - threshold):
                at_limits_all += 1
        r_saturation = -5.0 * (at_limits_all / 12.0)
        
        # 8. Stuck penalty - penalize when ACTIVE joints stop moving
        #    Only check FLH, FRH (indices 0,1) + all vertical (indices 6-11)
        active_joints = np.concatenate([self._theta_cmd[0:2], self._theta_cmd[6:12]])
        active_joints_prev = np.concatenate([self._theta_cmd_prev[0:2], self._theta_cmd_prev[6:12]])
        active_change = np.abs(active_joints - active_joints_prev)
        mean_active_change = float(np.mean(active_change))
        r_stuck = -2.0 if mean_active_change < 0.001 else 0.0
        self._theta_cmd_prev = self._theta_cmd.copy()
        
        # 9. Tracking error penalty - closed-loop feedback (hardware only)
        if self.use_hardware and hasattr(self, '_theta_actual'):
            tracking_error = np.abs(self._theta_cmd - self._theta_actual)
            mean_tracking_error = float(np.mean(tracking_error))
            r_tracking = -3.0 * mean_tracking_error
        else:
            r_tracking = 0.0
        
        # Total reward: ULTRA-SIMPLE front-leg crawl (9 components)
        reward = (r_forward + r_upright + r_front_sync + r_front_oscillation + 
                  r_vertical + r_smooth + r_saturation + r_stuck + r_tracking)

        terminated = (abs(math.degrees(roll)) > self.fall_deg) or (abs(math.degrees(pitch)) > self.fall_deg)
        truncated  = (self._t >= self.steps_max)
        self._t += 1
        return obs, reward, terminated, truncated, {}

    def render(self): pass

    def close(self):
        if self.link is not None:
            self.link.close()
            self.link = None