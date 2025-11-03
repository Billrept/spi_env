from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hexapod_env import SPIBalance12Env

MODEL_PATH = "./runs/fixed_final/ppo_fixed"
VEC_PATH   = "./runs/fixed_final/vecnorm_fixed.pkl"
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
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            if done.any():
                obs = venv.reset()
    except KeyboardInterrupt:
        pass
    finally:
        venv.close()

if __name__ == "__main__":
    main()
