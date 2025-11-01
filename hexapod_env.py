"""
Hexapod Balance Environment for PPO Training
Simulates a 6-DOF hexapod with IMU feedback for self-balancing and stepping.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import time


class HexapodBalanceEnv(gym.Env):
    """
    Custom Gymnasium environment for hexapod balance control.
    
    Observation:
        - IMU: roll, pitch, yaw (rad), gyro xyz (rad/s), accel xyz (g)
        - 6 joint positions (rad) relative to neutral
        Total: 15 dimensions, normalized to ~[-1, 1]
    
    Action:
        - 6 position deltas (rad per step), clamped per step
    
    Reward:
        - Uprightness: -(roll^2 + pitch^2)
        - Smoothness: -λ1 * ||Δaction||^2
        - Joint velocity penalty: -λ2 * ||joint_vel||^2
        - Alive bonus per step
        - Early termination if roll/pitch exceeds thresholds
        - Optional stepping bonus (phase 2)
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 50}
    
    # Constants
    TICKS_PER_REV = 4096
    NEUTRAL_TICKS = np.array([2048, 2048, 2048, 2048, 2048, 2048])
    
    # Safety limits (radians) - adjust based on physical robot
    JOINT_LIMITS_RAD = np.array([
        [-np.pi/2, np.pi/2],   # Joint 1
        [-np.pi/2, np.pi/2],   # Joint 2
        [-np.pi/2, np.pi/2],   # Joint 3
        [-np.pi/2, np.pi/2],   # Joint 4
        [-np.pi/2, np.pi/2],   # Joint 5
        [-np.pi/2, np.pi/2],   # Joint 6
    ])
    
    # Reward weights
    REWARD_WEIGHTS = {
        "upright": 1.0,
        "smooth": 0.01,
        "joint_vel": 0.005,
        "alive": 0.5,
        "step_bonus": 0.05,
    }
    
    def __init__(
        self,
        max_episode_steps: int = 500,  # 10s at 50Hz
        fall_threshold_deg: float = 35.0,
        action_delta_clamp: float = 0.05,  # rad per step
        enable_stepping: bool = False,
        domain_randomization: bool = True,
        action_smoothing_alpha: float = 0.3,
        target_dt: float = 0.02,  # 50 Hz
    ):
        super().__init__()
        
        self.max_episode_steps = max_episode_steps
        self.fall_threshold_rad = np.deg2rad(fall_threshold_deg)
        self.action_delta_clamp = action_delta_clamp
        self.enable_stepping = enable_stepping
        self.domain_randomization = domain_randomization
        self.action_smoothing_alpha = action_smoothing_alpha
        self.target_dt = target_dt
        
        # Observation space: [roll, pitch, yaw, gyro_x, gyro_y, gyro_z, 
        #                     accel_x, accel_y, accel_z, joint_pos_1..6]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),
            dtype=np.float32
        )
        
        # Action space: 6 position deltas (rad)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        
        # Internal state
        self.step_count = 0
        self.episode_count = 0
        self.last_action = np.zeros(6)
        self.smoothed_action = np.zeros(6)
        self.last_step_time = None
        
        # Physics simulation state (simplified)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.gyro = np.zeros(3)
        self.accel = np.array([0.0, 0.0, 1.0])  # 1g down initially
        self.joint_positions = np.zeros(6)  # relative to neutral
        self.joint_velocities = np.zeros(6)
        
        # Domain randomization parameters
        self.imu_bias = np.zeros(9)
        self.action_scale_noise = 1.0
        self.action_delay_buffer = []
        
    def _get_obs(self) -> np.ndarray:
        """Get current observation with normalization."""
        # Apply IMU bias if domain randomization is enabled
        imu_data = np.array([
            self.roll, self.pitch, self.yaw,
            self.gyro[0], self.gyro[1], self.gyro[2],
            self.accel[0], self.accel[1], self.accel[2]
        ])
        
        if self.domain_randomization:
            imu_data += self.imu_bias
        
        # Wrap yaw to [-π, π]
        imu_data[2] = np.arctan2(np.sin(imu_data[2]), np.cos(imu_data[2]))
        
        obs = np.concatenate([imu_data, self.joint_positions]).astype(np.float32)
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary information."""
        return {
            "roll_deg": np.rad2deg(self.roll),
            "pitch_deg": np.rad2deg(self.pitch),
            "step_count": self.step_count,
            "episode_count": self.episode_count,
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to neutral standing posture."""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.episode_count += 1
        
        # Reset to neutral with small noise
        self.roll = self.np_random.uniform(-0.05, 0.05)
        self.pitch = self.np_random.uniform(-0.05, 0.05)
        self.yaw = self.np_random.uniform(-np.pi, np.pi)
        self.gyro = self.np_random.uniform(-0.1, 0.1, size=3)
        self.accel = np.array([0.0, 0.0, 1.0]) + self.np_random.uniform(-0.05, 0.05, size=3)
        
        # Joint positions with small noise
        self.joint_positions = self.np_random.uniform(-0.1, 0.1, size=6)
        self.joint_velocities = np.zeros(6)
        
        self.last_action = np.zeros(6)
        self.smoothed_action = np.zeros(6)
        self.last_step_time = time.time()
        
        # Domain randomization setup
        if self.domain_randomization:
            self.imu_bias = self.np_random.uniform(-0.02, 0.02, size=9)
            self.action_scale_noise = self.np_random.uniform(0.95, 1.05)
            self.action_delay_buffer = []
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.step_count += 1
        
        # Rate limiting (50 Hz target)
        if self.last_step_time is not None:
            elapsed = time.time() - self.last_step_time
            if elapsed < self.target_dt:
                time.sleep(self.target_dt - elapsed)
        self.last_step_time = time.time()
        
        # Clamp action deltas
        action = np.clip(action, -1.0, 1.0) * self.action_delta_clamp
        
        # Apply domain randomization
        if self.domain_randomization:
            action = action * self.action_scale_noise
            
            # Action delay jitter (±1 step)
            self.action_delay_buffer.append(action.copy())
            if len(self.action_delay_buffer) > 2:
                self.action_delay_buffer.pop(0)
            
            if self.np_random.random() < 0.1 and len(self.action_delay_buffer) > 1:
                action = self.action_delay_buffer[0]
            else:
                action = self.action_delay_buffer[-1]
        
        # Exponential smoothing
        self.smoothed_action = (
            self.action_smoothing_alpha * action +
            (1 - self.action_smoothing_alpha) * self.smoothed_action
        )
        
        # Update joint positions
        new_joint_positions = self.joint_positions + self.smoothed_action
        
        # Clamp to limits
        for i in range(6):
            new_joint_positions[i] = np.clip(
                new_joint_positions[i],
                self.JOINT_LIMITS_RAD[i, 0],
                self.JOINT_LIMITS_RAD[i, 1]
            )
        
        # Compute joint velocities (finite difference)
        self.joint_velocities = (new_joint_positions - self.joint_positions) / self.target_dt
        self.joint_positions = new_joint_positions
        
        # Simple physics simulation (placeholder - replace with proper dynamics)
        self._simulate_physics()
        
        # Compute reward
        reward = self._compute_reward(action)
        
        # Check termination conditions
        terminated = self._check_fall()
        truncated = self.step_count >= self.max_episode_steps
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _simulate_physics(self):
        """
        Simplified physics simulation.
        In reality, this would be replaced by actual robot dynamics or a physics engine.
        
        For now, we model a simple pendulum-like behavior influenced by joint positions.
        """
        # Compute torques based on joint configuration
        # Simplified: joint deviations create roll/pitch disturbances
        left_joints = self.joint_positions[:3]  # Assume first 3 are left side
        right_joints = self.joint_positions[3:]  # Last 3 are right side
        
        # Roll influenced by left-right asymmetry
        roll_torque = (np.sum(left_joints) - np.sum(right_joints)) * 0.1
        
        # Pitch influenced by front-back positioning
        pitch_torque = (np.sum(self.joint_positions[::2]) - np.sum(self.joint_positions[1::2])) * 0.1
        
        # Update angular velocities (simplified integration)
        dt = self.target_dt
        damping = 0.95
        
        self.gyro[0] += roll_torque * dt  # roll rate
        self.gyro[1] += pitch_torque * dt  # pitch rate
        self.gyro *= damping
        
        # Update orientation
        self.roll += self.gyro[0] * dt
        self.pitch += self.gyro[1] * dt
        
        # Gravity effect on accelerometer
        # In body frame: accel measures specific force
        self.accel[0] = -np.sin(self.pitch) + self.np_random.uniform(-0.02, 0.02)
        self.accel[1] = np.sin(self.roll) * np.cos(self.pitch) + self.np_random.uniform(-0.02, 0.02)
        self.accel[2] = np.cos(self.roll) * np.cos(self.pitch) + self.np_random.uniform(-0.02, 0.02)
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward based on current state and action."""
        W = self.REWARD_WEIGHTS
        
        # Uprightness: penalize deviation from upright
        r_upright = -(self.roll**2 + self.pitch**2) * W["upright"]
        
        # Smoothness: penalize large action changes
        action_diff = action - self.last_action
        r_smooth = -np.sum(action_diff**2) * W["smooth"]
        
        # Joint velocity penalty
        r_joint_vel = -np.sum(self.joint_velocities**2) * W["joint_vel"]
        
        # Alive bonus
        r_alive = W["alive"]
        
        # Total reward
        reward = r_upright + r_smooth + r_joint_vel + r_alive
        
        # Optional stepping bonus (phase 2)
        if self.enable_stepping:
            # Simple heuristic: reward alternating leg motion
            left_motion = np.sum(np.abs(self.joint_velocities[:3]))
            right_motion = np.sum(np.abs(self.joint_velocities[3:]))
            stepping_score = min(left_motion, right_motion)
            reward += stepping_score * W["step_bonus"]
        
        self.last_action = action.copy()
        
        return float(reward)
    
    def _check_fall(self) -> bool:
        """Check if robot has fallen."""
        return (
            abs(self.roll) > self.fall_threshold_rad or
            abs(self.pitch) > self.fall_threshold_rad
        )
    
    def render(self):
        """Render the environment (placeholder)."""
        if self.render_mode == "human":
            print(f"Step {self.step_count}: "
                  f"Roll={np.rad2deg(self.roll):.1f}°, "
                  f"Pitch={np.rad2deg(self.pitch):.1f}°")
    
    def close(self):
        """Clean up resources."""
        pass
    
    def set_enable_stepping(self, enable: bool):
        """Enable or disable stepping phase (for curriculum learning)."""
        self.enable_stepping = enable
        if enable:
            print(f"[Env {id(self)}] Stepping phase enabled!")
    
    @staticmethod
    def ticks_to_rad(ticks: np.ndarray) -> np.ndarray:
        """Convert encoder ticks to radians."""
        return (ticks / HexapodBalanceEnv.TICKS_PER_REV) * 2 * np.pi
    
    @staticmethod
    def rad_to_ticks(rad: np.ndarray) -> np.ndarray:
        """Convert radians to encoder ticks."""
        return (rad / (2 * np.pi) * HexapodBalanceEnv.TICKS_PER_REV).astype(int)
    
    @staticmethod
    def hundredths_to_rad(hundredths: np.ndarray) -> np.ndarray:
        """Convert hundredths of degrees to radians."""
        return np.deg2rad(hundredths / 100.0)
    
    @staticmethod
    def rad_to_hundredths(rad: np.ndarray) -> np.ndarray:
        """Convert radians to hundredths of degrees."""
        return (np.rad2deg(rad) * 100.0).astype(int)


# Vectorized environment wrapper for parallel training
def make_hexapod_env(rank: int, seed: int = 0, **kwargs):
    """
    Create a hexapod environment for parallel training.
    
    Args:
        rank: Environment ID for seeding
        seed: Base random seed
        **kwargs: Additional environment parameters
    """
    def _init():
        env = HexapodBalanceEnv(**kwargs)
        env.reset(seed=seed + rank)
        return env
    return _init
