from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from gymnasium.wrappers import RecordEpisodeStatistics
from hexapod_env import SPIBalance12Env
import numpy as np

# ===== SAFETY SETTINGS =====
SERIAL_PORT = "/dev/tty.usbserial-XXXX"  # CHANGE THIS to your port (COM3 on Windows)
MAX_TILT_SAFETY = 35.0  # degrees - emergency stop if robot tilts too much
EPISODE_SECONDS_SAFE = 5.0  # shorter episodes for safety during training
TOTAL_TIMESTEPS = 50_000  # fewer steps since hardware is slower

class SafetyCallback(BaseCallback):
    """Monitor training and log safety statistics"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_tilts = []
        self.emergency_stops = 0
        
    def _on_step(self) -> bool:
        # Check if episode ended due to safety
        if len(self.locals.get('dones', [])) > 0:
            if self.locals['dones'][0]:
                # Log episode statistics
                info = self.locals.get('infos', [{}])[0]
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                    print(f"Episode: reward={ep_reward:.2f}, length={ep_length}")
        
        # Emergency stop check (optional - can monitor external kill switch)
        # if check_emergency_button():
        #     print("EMERGENCY STOP TRIGGERED")
        #     return False
        
        return True
    
    def _on_training_end(self) -> None:
        print(f"\n=== Training Complete ===")
        print(f"Emergency stops: {self.emergency_stops}")

def make_hardware_env():
    """Create environment with safety wrapper"""
    base_env = SPIBalance12Env(
        use_hardware=True, 
        port=SERIAL_PORT,
        episode_seconds=EPISODE_SECONDS_SAFE,
        max_tilt_deg=MAX_TILT_SAFETY  # Will need to add this parameter
    )
    return RecordEpisodeStatistics(base_env)

def main():
    print("=" * 60)
    print("HARDWARE TRAINING MODE")
    print("=" * 60)
    print(f"Port: {SERIAL_PORT}")
    print(f"Safety tilt limit: {MAX_TILT_SAFETY}°")
    print(f"Episode duration: {EPISODE_SECONDS_SAFE}s")
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")
    print("=" * 60)
    print("\n⚠️  SAFETY CHECKLIST:")
    print("  [ ] Robot on soft surface or suspended")
    print("  [ ] Emergency stop button within reach")
    print("  [ ] Joint limits verified in textimu_link.py")
    print("  [ ] IMU data streaming correctly")
    print("  [ ] Battery level sufficient")
    print("\nPress ENTER to start training, or Ctrl+C to abort...")
    input()
    
    # --- Train env with VecNormalize ---
    train_venv = DummyVecEnv([make_hardware_env])
    train_venv = VecNormalize(
        train_venv, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.0,
        gamma=0.99
    )
    
    # --- PPO with conservative settings for hardware ---
    model = PPO(
        "MlpPolicy", 
        train_venv,
        learning_rate=1e-4,        # Lower learning rate for stability
        n_steps=512,               # Smaller rollout buffer (hardware is slower)
        batch_size=64,             # Smaller batches
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,            # More conservative clipping
        ent_coef=0.005,            # Less random exploration (safety)
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tb_hardware/"
    )
    
    # --- Callbacks ---
    safety_cb = SafetyCallback()
    checkpoint_cb = CheckpointCallback(
        save_freq=5_000,  # Save more frequently
        save_path="./runs/hardware_ckpt/",
        name_prefix="ppo_hardware"
    )
    
    # --- Train ---
    print("\n🚀 Starting hardware training...\n")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[safety_cb, checkpoint_cb],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    finally:
        # Save final model
        print("\n💾 Saving final model...")
        model.save("./runs/hardware_final/ppo_hardware")
        train_venv.save("./runs/hardware_final/vecnorm_hardware.pkl")
        print("✅ Model saved successfully")
        
        # Close environment (stops sending commands to robot)
        train_venv.close()
        print("✅ Robot connection closed")

if __name__ == "__main__":
    main()
