import numpy as np
from hexapod_env import SPIBalance12Env
from stable_baselines3 import PPO

def analyze_trained_policy(model_path="./runs/best/best_model.zip", steps=500):
    """Analyze what movements the policy learned"""
    print("=" * 70)
    print("Analyzing Trained Policy Behavior")
    print("=" * 70)
    
    # Load model
    model = PPO.load(model_path)
    print(f"✓ Loaded model: {model_path}\n")
    
    # Create environment
    env = SPIBalance12Env(use_hardware=False, seed=42)
    obs, _ = env.reset()
    
    # Track metrics
    positions = []
    velocities = []
    actions_history = []
    joint_angles_H = [[] for _ in range(6)]
    joint_angles_V = [[] for _ in range(6)]
    rewards_breakdown = []
    
    total_reward = 0
    
    print("Running policy for 500 steps...\n")
    for t in range(steps):
        # Get action from policy
        action, _ = model.predict(obs, deterministic=True)
        
        # Track action
        actions_history.append(action.copy())
        
        # Step environment
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        
        # Track position
        positions.append(env._position_x)
        velocities.append(getattr(env, '_velocity_x', 0.0))
        
        # Track joint angles
        for i in range(6):
            joint_angles_H[i].append(obs[9 + i])  # Horizontal
            joint_angles_V[i].append(obs[15 + i])  # Vertical
        
        # Calculate reward breakdown (same as env)
        roll, pitch = obs[0], obs[1]
        qH = obs[9:15]
        qV = obs[15:21]
        
        forward_distance = env._position_x - env._last_position_x
        r_forward = 1000.0 * forward_distance  # Updated to match new reward scale
        
        import math
        angle_mag = math.sqrt(roll**2 + pitch**2)
        r_upright = 0.1 * math.exp(-2.5 * angle_mag)  # Updated scale
        
        r_smooth = -0.001 * float(np.sum(action**2))  # Updated scale
        
        qH_variance = float(np.var(qH))
        r_gait = 0.05 * min(qH_variance, 0.4)  # Updated scale
        
        pair_opposites = -(qH[0] * qH[1]) - (qH[2] * qH[3]) - (qH[4] * qH[5])
        r_pair_coordination = 0.02 * np.clip(pair_opposites, -0.5, 0.5)  # Updated scale
        
        r_vertical_penalty = -0.001 * float(np.sum(qV**2))  # Updated scale
        
        rewards_breakdown.append({
            'forward': r_forward,
            'upright': r_upright,
            'smooth': r_smooth,
            'gait': r_gait,
            'pair': r_pair_coordination,
            'vertical': r_vertical_penalty,
            'total': reward
        })
        
        if done or truncated:
            print(f"Episode ended at step {t}")
            break
    
    # Analysis
    print("=" * 70)
    print("RESULTS:")
    print("=" * 70)
    
    # Position analysis
    final_position = positions[-1]
    max_position = max(positions)
    min_position = min(positions)
    
    print(f"\n📍 POSITION TRACKING:")
    print(f"  Final position:     {final_position:.4f} m ({final_position*100:.2f} cm)")
    print(f"  Max position:       {max_position:.4f} m ({max_position*100:.2f} cm)")
    print(f"  Min position:       {min_position:.4f} m ({min_position*100:.2f} cm)")
    print(f"  Net movement:       {final_position:.4f} m ({final_position*100:.2f} cm)")
    
    # Velocity analysis
    avg_velocity = np.mean(velocities)
    max_velocity = max(velocities)
    
    print(f"\n🚀 VELOCITY:")
    print(f"  Average velocity:   {avg_velocity:.4f} m/s ({avg_velocity*100:.2f} cm/s)")
    print(f"  Max velocity:       {max_velocity:.4f} m/s ({max_velocity*100:.2f} cm/s)")
    
    # Action analysis
    actions_array = np.array(actions_history)
    action_means = np.mean(np.abs(actions_array), axis=0)
    
    print(f"\n🎮 ACTION STATISTICS:")
    print(f"  Horizontal actions (avg magnitude):")
    for i in range(6):
        print(f"    Leg {i}: {action_means[i]:.4f} rad")
    print(f"  Vertical actions (avg magnitude):")
    for i in range(6):
        print(f"    Leg {i}: {action_means[6+i]:.4f} rad")
    
    # Joint angle patterns
    print(f"\n🦿 JOINT ANGLE PATTERNS:")
    for i in range(6):
        h_range = max(joint_angles_H[i]) - min(joint_angles_H[i])
        v_range = max(joint_angles_V[i]) - min(joint_angles_V[i])
        print(f"  Leg {i} - H range: {h_range:.3f} rad ({np.degrees(h_range):.1f}°), "
              f"V range: {v_range:.3f} rad ({np.degrees(v_range):.1f}°)")
    
    # Reward breakdown
    avg_rewards = {
        'forward': np.mean([r['forward'] for r in rewards_breakdown]),
        'upright': np.mean([r['upright'] for r in rewards_breakdown]),
        'smooth': np.mean([r['smooth'] for r in rewards_breakdown]),
        'gait': np.mean([r['gait'] for r in rewards_breakdown]),
        'pair': np.mean([r['pair'] for r in rewards_breakdown]),
        'vertical': np.mean([r['vertical'] for r in rewards_breakdown]),
    }
    
    print(f"\n💰 REWARD BREAKDOWN (average per step):")
    print(f"  Forward progress:   {avg_rewards['forward']:+.4f}")
    print(f"  Upright stability:  {avg_rewards['upright']:+.4f}")
    print(f"  Smooth actions:     {avg_rewards['smooth']:+.4f}")
    print(f"  Gait coordination:  {avg_rewards['gait']:+.4f}")
    print(f"  Pair coordination:  {avg_rewards['pair']:+.4f}")
    print(f"  Vertical penalty:   {avg_rewards['vertical']:+.4f}")
    print(f"  ─────────────────────────────────")
    print(f"  Average per step:   {total_reward/len(positions):+.4f}")
    print(f"  Total reward:       {total_reward:.2f}")
    
    # Diagnosis
    print(f"\n" + "=" * 70)
    print("DIAGNOSIS:")
    print("=" * 70)
    
    if abs(final_position) < 0.01:
        print("❌ PROBLEM: Robot is NOT moving forward!")
        print("   → Final position < 1cm")
        print("   → Policy learned to balance, not walk")
        print("   → Forward reward component is ~0")
    elif final_position < 0.1:
        print("⚠️  WARNING: Minimal forward movement")
        print("   → Robot moves but very slowly (<10cm in 10 seconds)")
        print("   → Physics might be too weak")
    else:
        print("✅ SUCCESS: Robot is moving forward!")
        print(f"   → Traveled {final_position*100:.1f}cm in 10 seconds")
    
    if avg_velocity < 0.001:
        print("\n❌ VELOCITY PROBLEM: Average velocity near zero!")
        print("   → Physics simulation may not be working correctly")
        print("   → Check _velocity_x calculation in _imu_offline()")
    
    if np.mean(action_means[:6]) < 0.005:
        print("\n⚠️  EXPLORATION: Horizontal actions are very small")
        print("   → Policy isn't using horizontal joints much")
        print("   → May need more exploration (higher entropy coefficient)")
    
    print("\n" + "=" * 70)
    print("\nNext steps:")
    print("1. Check if _position_x is actually being updated in physics")
    print("2. Verify forward_force calculation produces non-zero values")
    print("3. Consider increasing physics multipliers or reducing DRAG")
    print("4. May need to reset and retrain with stronger physics")


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./runs/best/best_model.zip"
    analyze_trained_policy(model_path)
