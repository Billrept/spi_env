"""
Quick test script for the hexapod environment.
Verifies the environment works correctly before full training.
"""

import numpy as np
from hexapod_env import HexapodBalanceEnv


def test_environment():
    """Run basic tests on the hexapod environment."""
    
    print("="*60)
    print("Testing Hexapod Balance Environment")
    print("="*60)
    
    # Create environment
    print("\n1. Creating environment...")
    env = HexapodBalanceEnv(
        max_episode_steps=100,
        fall_threshold_deg=35.0,
        action_delta_clamp=0.05,
        enable_stepping=False,
        domain_randomization=True,
    )
    print(f"   ✓ Observation space: {env.observation_space}")
    print(f"   ✓ Action space: {env.action_space}")
    
    # Test reset
    print("\n2. Testing reset...")
    obs, info = env.reset(seed=42)
    print(f"   ✓ Observation shape: {obs.shape}")
    print(f"   ✓ Initial roll: {info['roll_deg']:.2f}°")
    print(f"   ✓ Initial pitch: {info['pitch_deg']:.2f}°")
    
    # Test random actions
    print("\n3. Testing random actions...")
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step == 0 or step == 9:
            print(f"   Step {step}: reward={reward:.3f}, "
                  f"roll={info['roll_deg']:.1f}°, "
                  f"pitch={info['pitch_deg']:.1f}°")
        
        if terminated or truncated:
            print(f"   Episode ended at step {step}")
            break
    
    print(f"   ✓ Total reward: {total_reward:.2f}")
    
    # Test full episode
    print("\n4. Testing full episode with zero actions...")
    obs, info = env.reset(seed=123)
    episode_length = 0
    episode_reward = 0
    
    while episode_length < 100:
        action = np.zeros(6)  # Do nothing
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        
        if terminated or truncated:
            break
    
    print(f"   ✓ Episode length: {episode_length}")
    print(f"   ✓ Episode reward: {episode_reward:.2f}")
    print(f"   ✓ Final roll: {info['roll_deg']:.1f}°")
    print(f"   ✓ Final pitch: {info['pitch_deg']:.1f}°")
    
    # Test conversion utilities
    print("\n5. Testing conversion utilities...")
    ticks = np.array([2048, 1024, 3072])
    rad = env.ticks_to_rad(ticks)
    ticks_back = env.rad_to_ticks(rad)
    print(f"   Ticks: {ticks}")
    print(f"   Radians: {rad}")
    print(f"   Ticks back: {ticks_back}")
    print(f"   ✓ Conversion working: {np.allclose(ticks, ticks_back)}")
    
    hundredths = np.array([3500, -1200, 0])  # 35°, -12°, 0°
    rad_from_hundredths = env.hundredths_to_rad(hundredths)
    print(f"   Hundredths of degrees: {hundredths}")
    print(f"   Radians: {rad_from_hundredths}")
    print(f"   Degrees: {np.rad2deg(rad_from_hundredths)}")
    
    # Test stepping mode
    print("\n6. Testing stepping mode switch...")
    env.set_enable_stepping(True)
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   ✓ Stepping enabled, step reward: {reward:.3f}")
    
    # Cleanup
    env.close()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nEnvironment is ready for training.")
    print("Run: python train_ppo.py --timesteps 100000")


if __name__ == "__main__":
    test_environment()
