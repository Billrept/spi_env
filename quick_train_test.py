from hexapod_env import SPIBalance12Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import os

print("=" * 70)
print("QUICK TRAINING TEST - Improved Reward Structure")
print("=" * 70)
print("\nChanges:")
print("  • Forward reward: 15x → 1000x (massively increased!)")
print("  • Physics boost: 1.5x → 3.0x force multiplier")
print("  • Reduced MASS: 1.8 → 1.5 kg (more responsive)")
print("  • Increased FRICTION: 0.75 → 0.85 (better grip)")
print("  • Increased DRAG: 0.88 → 0.92 (more momentum)")
print("\n" + "=" * 70 + "\n")

# Create environment
env = SPIBalance12Env(use_hardware=False, seed=42)
eval_env = SPIBalance12Env(use_hardware=False, seed=123)

# Setup evaluation callback
os.makedirs("./runs/test", exist_ok=True)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./runs/test",
    log_path="./runs/test",
    eval_freq=5000,
    deterministic=True,
    render=False,
    n_eval_episodes=5
)

# Create PPO model with same hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    ent_coef=0.01,  # Exploration
    verbose=1,
    tensorboard_log="./tb"
)

print("Training for 50,000 timesteps...")
print("Watch for forward reward component to become non-zero!\n")

model.learn(
    total_timesteps=50_000,
    callback=eval_callback,
    progress_bar=True
)

print("\n" + "=" * 70)
print("Quick test complete!")
print("=" * 70)
print("\nCheck results:")
print("  • Model saved to: ./runs/test/best_model.zip")
print("  • Run: python analyze_policy.py ./runs/test/best_model.zip")
print("  • Look for forward reward > 0.01 per step")
print("\nIf forward reward is still ~0, the physics may need more tuning.")
