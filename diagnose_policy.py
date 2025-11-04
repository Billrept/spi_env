from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hexapod_env import SPIBalance12Env
import numpy as np

def diagnose():
    print("=" * 70)
    print("POLICY DIAGNOSTIC TOOL")
    print("=" * 70)
    
    # Load model and vecnorm
    print("\n1. Loading model and VecNormalize...")
    try:
        env = DummyVecEnv([lambda: SPIBalance12Env(use_hardware=False)])
        env = VecNormalize.load('./runs/final/vecnorm.pkl', env)
        env.training = False
        model = PPO.load('./runs/final/ppo_spi12_offline', env=env)
        print("   ✓ Model and VecNormalize loaded successfully")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return
    
    # Check VecNormalize statistics
    print("\n2. VecNormalize Statistics:")
    print(f"   theta_cmd mean: {env.obs_rms.mean[9:]}")
    print(f"   theta_cmd std:  {np.sqrt(env.obs_rms.var[9:])}")
    
    # Test reset
    print("\n3. Testing Environment Reset...")
    obs_raw = env.reset()
    print(f"   Normalized observation shape: {obs_raw.shape}")
    print(f"   Normalized theta_cmd: {obs_raw[0, 9:]}")
    print(f"   Range: [{obs_raw[0, 9:].min():.4f}, {obs_raw[0, 9:].max():.4f}]")
    
    # Get un-normalized observation
    base_env = env.venv.envs[0]
    theta_actual = base_env._theta_cmd
    print(f"\n   Actual (un-normalized) theta_cmd: {theta_actual}")
    print(f"   Range: [{theta_actual.min():.4f}, {theta_actual.max():.4f}]")
    
    # Test policy predictions
    print("\n4. Testing Policy Actions (first 20 steps)...")
    obs = env.reset()
    
    for i in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        theta_actual = base_env._theta_cmd
        
        if i < 5 or i % 5 == 0:
            print(f"\n   Step {i:2d}:")
            print(f"     Action range: [{action[0].min():.4f}, {action[0].max():.4f}]")
            print(f"     Actual theta: [{theta_actual.min():.4f}, {theta_actual.max():.4f}]")
            print(f"     Reward: {reward[0]:.2f}")
        
        if done[0]:
            print(f"   Episode ended at step {i}")
            break
    
    # Check if theta_cmd is stuck at limits
    print("\n5. Checking for Stuck at Limits...")
    at_upper_limit = np.sum(np.abs(theta_actual - 0.785) < 0.01)
    at_lower_limit = np.sum(np.abs(theta_actual + 0.785) < 0.01)
    at_limits = at_upper_limit + at_lower_limit
    
    print(f"   Joints at +0.785 rad (+45°): {at_upper_limit}/12")
    print(f"   Joints at -0.785 rad (-45°): {at_lower_limit}/12")
    print(f"   Total at limits: {at_limits}/12")
    
    if at_limits > 6:
        print("   ⚠️  WARNING: More than half the joints are at limits!")
        print("   This suggests the policy is saturating.")
    
    # Conversion test
    print("\n6. Testing Servo Conversion...")
    test_angles = [0.0, 0.785, -0.785, 0.1, -0.1]
    for angle in test_angles:
        ticks = 2048 + int(round(angle * 651.74))
        degrees = angle * 180 / np.pi
        print(f"   {angle:+.3f} rad ({degrees:+6.2f}°) -> {ticks} ticks")
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if at_limits > 6:
        print("  1. Policy is saturating at joint limits")
        print("     → Check if vecnorm.pkl matches the model")
        print("     → Verify both computers use same files")
    else:
        print("  1. ✓ Policy appears to be working correctly in simulation")
        print("     → Issue may be hardware-specific")
        print("     → Check serial communication on deployment computer")

if __name__ == "__main__":
    diagnose()
