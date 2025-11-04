from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hexapod_env_fixed import SPIBalance12Env  # ✅ This is the FIXED one
import time
import numpy as np

MODEL_PATH = "./runs/fixed2_final/ppo_fixed2"
VEC_PATH   = "./runs/fixed2_final/vecnorm_fixed2.pkl"
SERIAL_PORT = "COM3"   # set your port
CONTROL_HZ = 50  # Match the training environment's 50Hz

def make_online_env():
    env = SPIBalance12Env(use_hardware=True, port=SERIAL_PORT)
    env.reset()  # Initialize hardware connection
    return env

def main():
    base = DummyVecEnv([make_online_env])
    venv = VecNormalize.load(VEC_PATH, base)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(MODEL_PATH, env=venv, device='cpu')  # Force CPU usage
    obs = venv.reset()
    
    print("\nStarting control loop at 50Hz...")
    print("Press Ctrl+C to stop\n")
    
    step_count = 0
    last_time = time.time()
    last_action = None
    
    try:
        while True:
            current_time = time.time()
            dt = current_time - last_time
            
            # Enforce 50Hz control rate
            if dt < 1.0/CONTROL_HZ:
                time.sleep(1.0/CONTROL_HZ - dt)
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            
            # Print status every second
            if step_count % CONTROL_HZ == 0:
                # Handle vectorized environment outputs
                reward_val = reward[0] if isinstance(reward, np.ndarray) else reward
                forward_dist = info[0].get('forward_dist', 0.0) if isinstance(info, list) else info.get('forward_dist', 0.0)
                print(f"[Episode {step_count//CONTROL_HZ}] Avg Reward: {reward_val:.2f}, Avg Forward: {forward_dist:.2f}cm")
            
            if done.any():
                print("\nEpisode finished - Resetting...")
                obs = venv.reset()
                step_count = 0
            
            step_count += 1
            last_time = current_time
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        venv.close()

if __name__ == "__main__":
    main()
