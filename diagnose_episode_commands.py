#!/usr/bin/env python3
"""
Diagnose episode-level command sending patterns.
Monitors what happens during reset vs step.
"""
from hexapod_env import SPIBalance12Env
import numpy as np
import time

def diagnose_episode_pattern(port="COM3"):
    """
    Monitor command sending during actual episode execution
    """
    print("=" * 70)
    print("EPISODE COMMAND PATTERN DIAGNOSTIC")
    print("=" * 70)
    print(f"\nConnecting to {port}...")
    
    env = SPIBalance12Env(use_hardware=True, port=port, episode_seconds=5.0)
    
    print("\n" + "="*70)
    print("EPISODE 1: Monitoring command timing")
    print("="*70)
    
    # Track timing
    reset_start = time.time()
    obs, info = env.reset()
    reset_time = time.time() - reset_start
    
    print(f"\n[RESET] Completed in {reset_time*1000:.1f}ms")
    print(f"        Initial theta_cmd: {obs[9:]}")
    print(f"        Range: [{obs[9:].min():.3f}, {obs[9:].max():.3f}]")
    
    # Run 20 steps with detailed timing
    print(f"\n{'Step':<6} {'Action':<40} {'Time(ms)':<10} {'Cumulative(ms)'}")
    print("-" * 70)
    
    episode_start = time.time()
    prev_time = episode_start
    
    for step in range(20):
        # Simple oscillating action
        if step < 10:
            action = np.array([0.05, -0.05] * 6, dtype=np.float32)
        else:
            action = np.array([-0.05, 0.05] * 6, dtype=np.float32)
        
        step_start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - step_start
        cumulative = time.time() - episode_start
        
        action_str = f"[{action[0]:+.2f}, {action[1]:+.2f}, ...]"
        print(f"{step:<6} {action_str:<40} {step_time*1000:<10.2f} {cumulative*1000:.2f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break
        
        # Small delay to match 50Hz control
        time.sleep(0.02 - step_time if step_time < 0.02 else 0)
    
    print(f"\n{'='*70}")
    print("EPISODE 2: Same pattern to check consistency")
    print("="*70)
    
    reset_start = time.time()
    obs, info = env.reset()
    reset_time = time.time() - reset_start
    
    print(f"\n[RESET] Completed in {reset_time*1000:.1f}ms")
    
    episode_start = time.time()
    
    for step in range(20):
        if step < 10:
            action = np.array([0.05, -0.05] * 6, dtype=np.float32)
        else:
            action = np.array([-0.05, 0.05] * 6, dtype=np.float32)
        
        step_start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - step_start
        cumulative = time.time() - episode_start
        
        if step % 5 == 0:
            print(f"Step {step}: {step_time*1000:.2f}ms, cumulative {cumulative*1000:.2f}ms")
        
        if terminated or truncated:
            break
        
        time.sleep(0.02 - step_time if step_time < 0.02 else 0)
    
    # Test with longer delays between commands
    print(f"\n{'='*70}")
    print("EPISODE 3: Testing with 50ms delays between steps")
    print("="*70)
    
    obs, info = env.reset()
    
    for step in range(10):
        action = np.array([0.05, -0.05] * 6, dtype=np.float32)
        
        step_start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - step_start
        
        print(f"Step {step}: Send time {step_time*1000:.2f}ms")
        
        # Long delay
        time.sleep(0.05)
        
        if terminated or truncated:
            break
    
    env.close()
    
    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    print("\nQUESTIONS TO ANSWER:")
    print("  1. Does reset() trigger movement? (First command)")
    print("  2. Do subsequent steps() trigger movement?")
    print("  3. Is there a timing difference between reset and step?")
    print("  4. Does adding delays help?")

if __name__ == "__main__":
    import sys
    
    port = sys.argv[1] if len(sys.argv) > 1 else "COM3"
    
    try:
        diagnose_episode_pattern(port)
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
