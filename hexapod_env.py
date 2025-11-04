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
from textimu_link import CH_MAP_12, INVERT_12, TICKS2RAD

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
        self._theta_actual = np.zeros(12, dtype=np.float32)  # Actual servo positions (hardware feedback)
        
        # obs: [roll, pitch, yaw, gx, gy, gz, ax, ay, az, theta_cmd(12), theta_actual(12), error(12)]
        # Hardware mode: 45-D (9 IMU + 12 cmd + 12 actual + 12 error)
        # Simulation mode: 21-D (9 IMU + 12 cmd)
        # Use larger space to accommodate both modes
        high_imu = np.array([math.pi]*3 + [50.0]*3 + [5.0]*3, dtype=np.float32)
        high_q   = np.array([0.785]*12, dtype=np.float32)  # ±45° joint limits
        high_error = np.array([1.57]*12, dtype=np.float32)  # Max tracking error ±90°
        self.observation_space = spaces.Box(
            low=-np.concatenate([high_imu, high_q, high_q, high_error]),
            high=np.concatenate([high_imu, high_q, high_q, high_error]),
            dtype=np.float32
        )

        # actions: 12 deltas (H first, then V)
        low  = np.array([-MAX_STEP_RAD_H]*6 + [-MAX_STEP_RAD_V]*6, dtype=np.float32)
        high = np.array([+MAX_STEP_RAD_H]*6 + [+MAX_STEP_RAD_V]*6, dtype=np.float32)
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
        """
        x = self._state.copy()
        qH = self._theta_cmd[:6]    # horizontal
        qV = self._theta_cmd[6:]    # vertical

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
        MASS = 1.5           # kg (lighter for more responsive movement, was 1.8)
        FRICTION = 0.85      # high grip for effective pushing (was 0.75)
        DRAG = 0.92          # more momentum preserved (was 0.88)
        MAX_SPEED = 0.8      # m/s (increased from 0.6 for faster locomotion)
        
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
            
            # Store for tracking error calculation
            self._theta_actual = theta_actual
            
            # Observation: [IMU(9), theta_cmd(12), theta_actual(12), tracking_error(12)]
            tracking_error = self._theta_cmd - theta_actual
            return np.array([roll, pitch, yaw, gx, gy, gz, ax, ay, az, 
                           *self._theta_cmd, *theta_actual, *tracking_error], dtype=np.float32)
        else:
            # Simulation mode or no servo feedback: assume perfect tracking
            # Use an explicit `theta_actual` variable (copy of commanded) so the
            # returned observation is clearer and avoids accidental duplication.
            # Observation: [IMU(9), theta_cmd(12), theta_actual(12)=theta_cmd, tracking_error(12)=0]
            # Must match 45-D observation space size
            theta_actual = self._theta_cmd.copy()
            zero_error = np.zeros(12, dtype=np.float32)
            return np.array([roll, pitch, yaw, gx, gy, gz, ax, ay, az,
                             *self._theta_cmd, *theta_actual, *zero_error], dtype=np.float32)

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
        action = np.clip(action, self.action_space.low, self.action_space.high)
        theta  = self._theta_cmd + action

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

        # ===== REWARD FUNCTION: Middle-Leg Gait Pattern =====
        # Goal: Learn to walk using middle legs (ML, MR) with pattern:
        # Lift → Swing Forward → Down → Push Back
        
        roll, pitch = obs[0], obs[1]
        qH = obs[9:15]   # horizontal joint angles [FL, FR, ML, MR, RL, RR]
        qV = obs[15:21]  # vertical joint angles [FLV, FRV, MLV, MRV, RLV, RRV]
        
        # Extract middle leg angles
        MLH, MRH = qH[2], qH[3]  # Middle horizontal (indices 2, 3)
        MLV, MRV = qV[2], qV[3]  # Middle vertical (indices 8, 9 in full observation)
        
        # 1. Forward progress reward (PRIMARY OBJECTIVE - DOMINANT REWARD)
        forward_distance = self._position_x - self._last_position_x
        r_forward = 1000.0 * forward_distance
        self._last_position_x = self._position_x
        
        # 2. Stability reward - stay upright
        angle_mag = math.sqrt(roll**2 + pitch**2)
        r_upright = 0.1 * math.exp(-2.5 * angle_mag)
        
        # 3. Middle leg coordination - encourage synchronized opposite movement
        #    MLH and MRH should move in opposite directions (±0.4 rad)
        middle_h_opposite = -(MLH * MRH)  # Negative product = opposite signs
        r_middle_coord = 0.5 * np.clip(middle_h_opposite, 0, 0.4)  # Reward up to 0.2
        
        # 4. Vertical lift reward - encourage lifting middle legs
        #    MLV and MRV should occasionally be positive (lifted)
        middle_v_lift = np.clip(MLV, 0, 0.3) + np.clip(MRV, 0, 0.3)
        r_lift = 0.3 * middle_v_lift  # Reward when legs are lifted
        
        # 5. Discourage other legs from moving too much
        #    Front and rear legs should stay mostly neutral
        other_h = np.concatenate([qH[:2], qH[4:]])  # FL, FR, RL, RR
        other_v = np.concatenate([qV[:2], qV[4:]])  # FLV, FRV, RLV, RRV
        r_other_penalty = -0.05 * (float(np.sum(other_h**2)) + float(np.sum(other_v**2)))
        
        # 6. Energy efficiency
        r_smooth = -0.001 * float(np.sum(action**2))
        
        # 7. Oscillation reward - encourage ACTIVE movement in middle legs
        #    Middle legs should be moving (changing position)
        theta_change_middle = np.abs(self._theta_cmd[2:4] - self._theta_cmd_prev[2:4])  # MLH, MRH
        r_oscillation = 0.3 * float(np.mean(theta_change_middle))
        self._theta_cmd_prev = self._theta_cmd.copy()
        
        # 8. STRONG saturation penalty - heavily penalize ANY joints at limits
        #    Check ALL 12 joints, not just middle legs
        at_limits_all = (np.abs(self._theta_cmd) > 0.75).sum()  # Count all saturated joints
        r_saturation = -5.0 * (at_limits_all / 12.0)  # Max penalty: -5.0 if all joints saturated
        
        # 9. Stuck penalty - penalize when joints stop moving entirely
        #    If mean change across ALL joints is near zero, heavily penalize
        theta_change_all = np.abs(self._theta_cmd - self._theta_cmd_prev)
        mean_change = float(np.mean(theta_change_all))
        r_stuck = -2.0 if mean_change < 0.001 else 0.0  # -2.0 if completely stuck
        
        # 10. Tracking error penalty - penalize when actual servo positions deviate from commanded
        #     This creates closed-loop feedback control (only available with hardware feedback)
        if self.use_hardware and hasattr(self, '_theta_actual'):
            tracking_error = np.abs(self._theta_cmd - self._theta_actual)
            mean_tracking_error = float(np.mean(tracking_error))
            # Penalize large tracking errors (servo can't keep up or is stuck)
            r_tracking = -3.0 * mean_tracking_error  # -3.0 if 1 radian error on average
        else:
            r_tracking = 0.0  # No feedback in simulation
        
        # Total reward: Forward movement + coordination + avoid saturation + feedback control
        reward = (r_forward + r_upright + r_middle_coord + r_lift + 
                  r_other_penalty + r_smooth + r_oscillation + r_saturation + r_stuck + r_tracking)

        terminated = (abs(math.degrees(roll)) > self.fall_deg) or (abs(math.degrees(pitch)) > self.fall_deg)
        truncated  = (self._t >= self.steps_max)
        self._t += 1
        return obs, reward, terminated, truncated, {}

    def render(self): pass

    def close(self):
        if self.link is not None:
            self.link.close()
            self.link = None
