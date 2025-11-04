import sys
print("Starting import...", flush=True)

from hexapod_env_fixed import SPIBalance12Env
print("Import successful!", flush=True)

import numpy as np
print("Creating environment...", flush=True)

env = SPIBalance12Env(use_hardware=False, seed=42)
print("Environment created!", flush=True)

print("Resetting...", flush=True)
obs, _ = env.reset()
print(f"Reset done. Position: {env._position_x}", flush=True)

print("\nTaking 100 steps with strong backward sweep...", flush=True)
for i in range(100):
    action = np.array([
        -0.1, 0.1, -0.1, 0.1, -0.1, 0.1,  # Horizontal: left negative, right positive
        -0.08, -0.08, -0.08, -0.08, -0.08, -0.08  # Vertical: all push down
    ], dtype=np.float32)
    
    obs, rew, done, trunc, _ = env.step(action)
    
    if i % 25 == 0:
        print(f"Step {i}: pos={env._position_x*100:.2f}cm, vel={env._velocity_x:.4f}m/s", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"FINAL POSITION: {env._position_x*100:.2f} cm", flush=True)
print(f"FINAL VELOCITY: {env._velocity_x:.4f} m/s", flush=True)
print(f"{'='*60}", flush=True)

if env._position_x > 0.05:
    print("✅ SUCCESS! Moving FORWARD - ready to train!", flush=True)
elif env._position_x > 0:
    print("⚠️  Moving forward but weak - increase FORCE_MULTIPLIER", flush=True)
elif env._position_x < -0.01:
    print("❌ MOVING BACKWARD - force direction is still wrong!", flush=True)
else:
    print("❌ NOT MOVING - physics broken!", flush=True)