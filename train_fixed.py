"""
FIXED TRAINING SCRIPT - With proper distance tracking through VecNormalize
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from gymnasium.wrappers import RecordEpisodeStatistics
from hexapod_env_fixed import SPIBalance12Env
import numpy as np
import torch
import gymnasium as gym

class DistanceTrackingWrapper(gym.Wrapper):
    """Wrapper to track cumulative distance in episode info"""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Access the base environment through wrapper chain
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        # Add cumulative distance to info so it survives VecNormalize
        if hasattr(base_env, '_cumulative_distance'):
            info['cumulative_distance'] = float(base_env._cumulative_distance)
            info['position_x'] = float(base_env._position_x)
            info['velocity_x'] = float(base_env._velocity_x)
        return obs, reward, terminated, truncated, info

class DiagnosticCallback(BaseCallback):
    """Log detailed metrics INCLUDING forward distance"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.forward_distances = []
        self.final_positions = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Check for episode end
        if 'dones' in self.locals and len(self.locals['dones']) > 0:
            if self.locals['dones'][0]:
                self.episode_count += 1
                info = self.locals.get('infos', [{}])[0]
                
                # Get episode stats
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                
                # CRITICAL: Get distance from info (passed through wrapper)
                if 'cumulative_distance' in info:
                    dist = info['cumulative_distance']
                    pos = info.get('position_x', 0)
                    vel = info.get('velocity_x', 0)
                    self.forward_distances.append(dist)
                    self.final_positions.append(pos)
                    
                    # Print every 10 episodes
                    if self.episode_count % 10 == 0:
                        avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                        avg_dist = np.mean(self.forward_distances[-10:]) if self.forward_distances else 0
                        avg_pos = np.mean(self.final_positions[-10:]) if self.final_positions else 0
                        print(f"\n[Episode {self.episode_count}] "
                              f"Reward: {avg_reward:.1f}, "
                              f"Distance: {avg_dist*100:.2f}cm, "
                              f"Final pos: {avg_pos*100:.2f}cm, "
                              f"Last vel: {vel:.3f}m/s")
        
        return True
    
    def _on_training_end(self) -> None:
        if self.forward_distances:
            print(f"\n{'='*70}")
            print(f"TRAINING SUMMARY")
            print(f"{'='*70}")
            print(f"Total episodes: {len(self.forward_distances)}")
            print(f"Avg distance (last 50): {np.mean(self.forward_distances[-50:])*100:.2f} cm")
            print(f"Max distance: {np.max(self.forward_distances)*100:.2f} cm")
            print(f"Avg final position (last 50): {np.mean(self.final_positions[-50:])*100:.2f} cm")

def env_fn():
    base_env = SPIBalance12Env(use_hardware=False)
    # Wrap with distance tracker BEFORE RecordEpisodeStatistics
    tracked_env = DistanceTrackingWrapper(base_env)
    return RecordEpisodeStatistics(tracked_env)

def main():
    print("=" * 70)
    print("FIXED HEXAPOD TRAINING - WITH DISTANCE TRACKING")
    print("=" * 70)
    print("\nKey changes:")
    print("  ✓ Force multiplier: 100.0 (was 10.0)")
    print("  ✓ Reward only for FORWARD movement (no backward)")
    print("  ✓ Distance tracking through VecNormalize")
    print("  ✓ Debug prints every 50 steps in environment")
    print("  ✓ Lower entropy coefficient (0.005) for stability")
    print("=" * 70 + "\n")

    # Create environments with proper wrapper order
    train_venv = DummyVecEnv([env_fn])
    train_venv = VecNormalize(
        train_venv, 
        norm_obs=True,
        norm_reward=False,  # Keep raw rewards
        clip_obs=10.0,
        gamma=0.99
    )

    eval_venv = DummyVecEnv([env_fn])
    eval_venv = VecNormalize(
        eval_venv, 
        norm_obs=True, 
        norm_reward=False, 
        clip_obs=10.0,
        training=False,
        gamma=0.99
    )

    # PPO with LOWER entropy (policy was too random before)
    model = PPO(
        "MlpPolicy",
        train_venv,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,  # LOWER entropy (was 0.02) - policy needs to converge!
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tb_fixed2/",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUsing device: {model.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70 + "\n")

    # Callbacks
    diagnostic_cb = DiagnosticCallback()
    
    eval_cb = EvalCallback(
        eval_venv,
        best_model_save_path="./runs/fixed2_best/",
        log_path="./runs/fixed2_eval/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path="./runs/fixed2_ckpt/",
        name_prefix="ppo_fixed2"
    )

    # Sync eval env stats
    eval_venv.obs_rms = train_venv.obs_rms
    eval_venv.ret_rms = train_venv.ret_rms

    print("Starting training for 200k steps (shorter to test faster)...")
    print("You should see debug prints from the environment every 50 steps!")
    print("Watch the 'Distance' and 'Final pos' values in episode summaries.\n")

    # Train
    model.learn(
        total_timesteps=200_000,  # Shorter for faster testing
        callback=[diagnostic_cb, eval_cb, checkpoint_cb],
        progress_bar=True
    )

    # Save
    print("\nSaving final model...")
    model.save("./runs/fixed2_final/ppo_fixed2")
    train_venv.save("./runs/fixed2_final/vecnorm_fixed2.pkl")

    train_venv.close()
    eval_venv.close()

if __name__ == "__main__":
    main()