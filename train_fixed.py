"""
FIXED TRAINING SCRIPT
Key improvements:
1. Higher entropy for more exploration
2. Curriculum learning approach
3. Better normalization settings
4. Diagnostic logging
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from gymnasium.wrappers import RecordEpisodeStatistics
from hexapod_env_fixed import SPIBalance12Env
import numpy as np

class DiagnosticCallback(BaseCallback):
    """Log detailed metrics to diagnose training"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.forward_distances = []
        
    def _on_step(self) -> bool:
        # Log every episode
        if len(self.locals.get('dones', [])) > 0:
            if self.locals['dones'][0]:
                info = self.locals.get('infos', [{}])[0]
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    
                    # Try to get environment to check forward distance
                    env = self.training_env.envs[0]
                    if hasattr(env, 'env'):
                        base_env = env.env
                        if hasattr(base_env, '_cumulative_distance'):
                            dist = base_env._cumulative_distance
                            self.forward_distances.append(dist)
                            
                            # Print progress every 10 episodes
                            if len(self.episode_rewards) % 10 == 0:
                                avg_reward = np.mean(self.episode_rewards[-10:])
                                avg_dist = np.mean(self.forward_distances[-10:]) if self.forward_distances else 0
                                print(f"\n[Episode {len(self.episode_rewards)}] "
                                      f"Avg Reward: {avg_reward:.2f}, "
                                      f"Avg Forward: {avg_dist*100:.2f}cm")
        
        return True

def env_fn():
    return RecordEpisodeStatistics(SPIBalance12Env(use_hardware=False))

def main():
    print("=" * 70)
    print("FIXED HEXAPOD TRAINING")
    print("=" * 70)
    print("\nKey Fixes:")
    print("  ✓ Velocity reset in reset()")
    print("  ✓ Simplified physics with 10x force multiplier")
    print("  ✓ Velocity-based reward (50x) + distance bonus (200x)")
    print("  ✓ Better initial pose (legs slightly down)")
    print("  ✓ Larger action space (0.1 rad H, 0.08 rad V)")
    print("  ✓ High entropy coefficient (0.02) for exploration")
    print("  ✓ Reward normalization DISABLED")
    print("=" * 70 + "\n")

    # Create environments
    train_venv = DummyVecEnv([env_fn])
    train_venv = VecNormalize(
        train_venv, 
        norm_obs=True,       # Normalize observations
        norm_reward=False,   # DON'T normalize our carefully tuned rewards!
        clip_obs=10.0,
        clip_reward=10000.0  # Allow large rewards
    )

    eval_venv = DummyVecEnv([env_fn])
    eval_venv = VecNormalize(
        eval_venv, 
        norm_obs=True, 
        norm_reward=False, 
        clip_obs=10.0,
        training=False
    )

    # PPO with HIGH exploration
    model = PPO(
        "MlpPolicy",
        train_venv,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # HIGH entropy for exploration!
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tb_fixed/"
    )

    # Callbacks
    diagnostic_cb = DiagnosticCallback()
    
    eval_cb = EvalCallback(
        eval_venv,
        best_model_save_path="./runs/fixed_best/",
        log_path="./runs/fixed_eval/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path="./runs/fixed_ckpt/",
        name_prefix="ppo_fixed"
    )

    # Sync eval env stats
    eval_venv.obs_rms = train_venv.obs_rms
    eval_venv.ret_rms = train_venv.ret_rms

    print("Starting training for 500k steps...")
    print("Watch for 'Avg Forward' to increase above 1cm!\n")

    # Train
    model.learn(
        total_timesteps=500_000,
        callback=[diagnostic_cb, eval_cb, checkpoint_cb],
        progress_bar=True
    )

    # Save
    print("\nSaving final model...")
    model.save("./runs/fixed_final/ppo_fixed")
    train_venv.save("./runs/fixed_final/vecnorm_fixed.pkl")

    # Final statistics
    if diagnostic_cb.forward_distances:
        final_avg = np.mean(diagnostic_cb.forward_distances[-50:])
        print(f"\n" + "=" * 70)
        print(f"TRAINING COMPLETE")
        print(f"=" * 70)
        print(f"Final average forward distance: {final_avg*100:.2f} cm")
        if final_avg > 0.05:
            print("✅ SUCCESS! Robot learned to move forward!")
        elif final_avg > 0.01:
            print("⚠️  Partial success - robot moves but slowly")
        else:
            print("❌ Still not moving - may need more training or physics tuning")
        print(f"=" * 70)

    train_venv.close()
    eval_venv.close()

if __name__ == "__main__":
    main()