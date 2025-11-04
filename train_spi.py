from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gymnasium.wrappers import RecordEpisodeStatistics
from hexapod_env import SPIBalance12Env

def env_fn():
    # Wrap order: (custom env) -> RecordEpisodeStatistics (optional) -> VecEnv (Dummy) -> VecNormalize
    return RecordEpisodeStatistics(SPIBalance12Env(use_hardware=False))

def main():
    # --- Train env ---
    train_venv = DummyVecEnv([env_fn])
    # Use VecNormalize for stable training (normalize observations, NOT rewards)
    # This is REQUIRED - the policy must be deployed with the same normalization!
    train_venv = VecNormalize(train_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Hyperparameters optimized for tripod gait (simpler coordination than middle-leg gait)
    model = PPO(
        "MlpPolicy", 
        train_venv, 
        learning_rate=5e-4,      # Increased: simpler reward landscape allows faster learning
        n_steps=2048,            # Keep: good balance for 10s episodes (500 steps)
        batch_size=64,           # Reduced: smaller batches = more frequent updates
        gamma=0.99,              # Keep: standard discount for locomotion
        gae_lambda=0.95,         # Keep: standard advantage estimation
        clip_range=0.2,          # Keep: standard PPO clipping
        ent_coef=0.05,           # Increased: more exploration to discover tripod pattern
        vf_coef=0.5,             # Standard value function coefficient
        max_grad_norm=0.5,       # Standard gradient clipping
        n_epochs=10,             # Standard number of epochs per update
        verbose=1, 
        tensorboard_log="./tb/"
    )

    # --- Eval env (must match training settings) ---
    eval_venv = DummyVecEnv([env_fn])
    eval_venv = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_venv.training = False                # don't update stats
    # sync running means/vars from train to eval
    eval_venv.obs_rms = train_venv.obs_rms
    eval_venv.ret_rms = train_venv.ret_rms

    eval_cb = EvalCallback(eval_venv, best_model_save_path="./runs/best/",
                           log_path="./runs/eval/", eval_freq=10_000,
                           deterministic=True, render=False)
    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path="./runs/ckpt/", name_prefix="ppo_tripod")

    # Increased timesteps for tripod gait convergence
    model.learn(total_timesteps=750_000, callback=[eval_cb, ckpt_cb])

    # Save model & normalization stats
    model.save("./runs/final/ppo_tripod_offline")
    train_venv.save("./runs/final/vecnorm.pkl")
    print("\n✅ Training complete! Model saved to ./runs/final/ppo_tripod_offline.zip")

    train_venv.close(); eval_venv.close()

if __name__ == "__main__":
    main()