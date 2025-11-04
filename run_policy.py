from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hexapod_env import SPIBalance12Env

MODEL_PATH = "./runs/final/ppo_front_crawl_offline"
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
    
    print("\n=== INITIAL OBSERVATION (21-D, NORMALIZED BY VecNormalize) ===")
    print(f"IMU (9D): {obs[0, :9]}")
    print(f"FLH, FRH commanded (2D): {obs[0, 9:11]}")
    print(f"All vertical (6D): {obs[0, 11:17]}")
    print(f"FLH, FRH actual (2D): {obs[0, 17:19]}")
    print(f"FLH, FRH tracking_error (2D): {obs[0, 19:21]}")
    print(f"Front H range: [{obs[0, 9:11].min():.4f}, {obs[0, 9:11].max():.4f}]")
    
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
                print(f"FLH, FRH commanded: {obs[0, 9:11]}")
                print(f"All vertical (6D): {obs[0, 11:17]}")
                print(f"FLH, FRH actual: {obs[0, 17:19]}")
                print(f"FLH, FRH error: {obs[0, 19:21]}")
                print(f"Front H range: [{obs[0, 9:11].min():.4f}, {obs[0, 9:11].max():.4f}]")
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
