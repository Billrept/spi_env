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
    # CRITICAL: norm_reward=False to preserve our carefully tuned reward structure!
    # The 1000x forward reward scaling is intentional and should not be normalized
    train_venv = VecNormalize(train_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO("MlpPolicy", train_venv, learning_rate=3e-4, n_steps=2048, batch_size=64,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, 
                verbose=1, tensorboard_log="./tb/")

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
    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path="./runs/ckpt/", name_prefix="ppo_spi12")

    model.learn(total_timesteps=500_000, callback=[eval_cb, ckpt_cb])

    # Save model & normalization stats
    model.save("./runs/final/ppo_spi12_offline")
    train_venv.save("./runs/final/vecnorm.pkl")

    train_venv.close(); eval_venv.close()

if __name__ == "__main__":
    main()
