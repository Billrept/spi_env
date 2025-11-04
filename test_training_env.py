from hexapod_env_fixed import SPIBalance12Env
import numpy as np

print("Testing if environment tracks distance during episode...")

# Create environment with fixed seed
env = SPIBalance12Env(use_hardware=False, seed=42)
obs, _ = env.reset()

print(f"Initial: pos={env._position_x:.4f}, cumulative={env._cumulative_distance:.4f}")

# Test with alternating forward/backward actions
for i in range(50):
    # Alternate between forward and backward movement
    if i % 2 == 0:
        action = np.ones(12, dtype=np.float32) * 0.05  # Forward
    else:
        action = np.ones(12, dtype=np.float32) * -0.05  # Backward
    
    obs, reward, done, truncated, info = env.step(action)
    
    if i % 10 == 0:
        print(f"Step {i}: pos={env._position_x:.4f}, "
              f"cumulative={env._cumulative_distance:.4f}, "
              f"reward={reward:.2f}, "
              f"dx={info.get('dx', 0.0):.4f}")

print(f"\nFinal position: {env._position_x:.4f}m ({env._position_x*100:.2f}cm)")
print(f"Total distance: {env._cumulative_distance:.4f}m ({env._cumulative_distance*100:.2f}cm)")

# Validation checks
is_tracking = env._cumulative_distance > abs(env._position_x)
print("\nValidation:")
print(f"✓ Total distance > |position|: {is_tracking}")
print(f"✓ Movement scale reasonable: {abs(env._position_x) < 0.5}")  # Should move less than 50cm
print(f"✓ Rewards given for forward motion: {info.get('total_reward', 0.0) > 0}")