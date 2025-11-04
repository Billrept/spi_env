from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hexapod_env import SPIBalance12Env

MODEL_PATH = "./runs/final/ppo_simple_offline"
VEC_PATH   = "./runs/final/vecnorm.pkl"
SERIAL_PORT = "COM3"   # set your port

def make_online_env():
    return SPIBalance12Env(use_hardware=True, port=SERIAL_PORT)

def main():
    base = DummyVecEnv([make_online_env])
    venv = VecNormalize.load(VEC_PATH, base)
    venv.training   = False
    venv.norm_reward = False

    model = PPO.load(MODEL_PATH, env=venv)
    obs = venv.reset()
    
    print("\n=== INITIAL OBSERVATION (NORMALIZED BY VecNormalize) ===")
    print(f"IMU (9D): {obs[0, :9]}")
    print(f"theta_cmd (12D): {obs[0, 9:21]}")
    print(f"theta_actual (12D): {obs[0, 21:33]}")
    print(f"tracking_error (12D): {obs[0, 33:45]}")
    print(f"theta_cmd range: [{obs[0, 9:21].min():.4f}, {obs[0, 9:21].max():.4f}]")
    
    step_count = 0
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            
            if step_count < 5 or step_count % 50 == 0:
                print(f"\n=== STEP {step_count} ===")
                print(f"Action: {action[0]}")
                print(f"Action range: [{action[0].min():.4f}, {action[0].max():.4f}]")
            
            obs, reward, done, info = venv.step(action)
            
            if step_count < 5 or step_count % 50 == 0:
                print(f"theta_cmd (12D): {obs[0, 9:21]}")
                print(f"theta_actual (12D): {obs[0, 21:33]}")
                print(f"tracking_error (12D): {obs[0, 33:45]}")
                print(f"theta_cmd range: [{obs[0, 9:21].min():.4f}, {obs[0, 9:21].max():.4f}]")
                print(f"Reward: {reward[0]:.4f}")
            
            step_count += 1
            
            if done.any():
                print(f"\n=== EPISODE DONE after {step_count} steps ===")
                obs = venv.reset()
                step_count = 0
    except KeyboardInterrupt:
        print(f"\nStopped after {step_count} steps")
    finally:
        venv.close()

if __name__ == "__main__":
    main()
