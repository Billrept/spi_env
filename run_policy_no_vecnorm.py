from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from hexapod_env import SPIBalance12Env

MODEL_PATH = "./runs/final/ppo_spi12_offline"
SERIAL_PORT = "COM3"   # set your port

def make_online_env():
    return SPIBalance12Env(use_hardware=True, port=SERIAL_PORT)

def main():
    # NO VecNormalize - use raw policy!
    venv = DummyVecEnv([make_online_env])
    model = PPO.load(MODEL_PATH, env=venv)
    
    obs = venv.reset()
    print("\n=== Starting policy deployment (NO VecNormalize) ===")
    print(f"Initial theta_cmd: {obs[0, 9:]}")
    print("Press Ctrl+C to stop\n")
    
    step_count = 0
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            
            if step_count % 50 == 0:
                theta_actual = venv.envs[0]._theta_cmd
                print(f"Step {step_count:4d}: theta range [{theta_actual.min():+.3f}, {theta_actual.max():+.3f}], reward {reward[0]:.2f}")
            
            step_count += 1
            
            if done.any():
                print(f"\n=== Episode ended at step {step_count} ===")
                obs = venv.reset()
                step_count = 0
    except KeyboardInterrupt:
        print(f"\n\nStopped after {step_count} steps")
    finally:
        venv.close()
        print("Connection closed.")

if __name__ == "__main__":
    main()
