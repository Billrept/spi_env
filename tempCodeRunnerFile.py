from hexapod_env_fixed import SPIBalance12Env
import numpy as np

print("="*70)
print("PHYSICS VERIFICATION TEST")
print("="*70)

env = SPIBalance12Env(use_hardware=False, seed=42)
obs, _ = env.reset()

print(f"\nInitial state:")
print(f"  Position: {env._position_x:.6f} m")
print(f"  Velocity: {env._velocity_x:.6f} m/s\n")

print("Applying strong actions for 100 steps...\n")

for step in range(100):
    action = np.array([
        -0.10, 0.10, -0.10, 0.10, -0.10, 0.10,
        -0.08, -0.08, -0.08, -0.08, -0.08, -0.08
    ], dtype=np.float32)
    
    obs, reward, done, truncated, _ = env.step(action)
    
    if step % 25 == 0:
        print(f"Step {step:3d}: pos={env._position_x*100:6.2f}cm, vel={env._velocity_x:7.4f}m/s, reward={reward:7.2f}")

print(f"\n" + "="*70)
print(f"FINAL POSITION: {env._position_x*100:.2f} cm")
print(f"FINAL VELOCITY: {env._velocity_x:.5f} m/s")
print("="*70)

if abs(env._position_x) < 0.001:
    print("\nCRITICAL: Position is 0! Physics is broken.")
elif abs(env._position_x) < 0.05:
    print("\nWARNING: Movement too small. Increase FORCE_MULTIPLIER.")
else:
    print(f"\nSUCCESS: Robot moved {env._position_x*100:.1f}cm!")