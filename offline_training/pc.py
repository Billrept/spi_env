# pc_collect.py
import serial
import numpy as np
import pickle
import time
import os
import threading

class RobotInterface:
    def __init__(self, port='COM3', baudrate=57600):
        self.ser = serial.Serial(port, baudrate, timeout=2)
        self.lock = threading.Lock()
        time.sleep(2)
        print(f"Connected to robot on {port}")
        
        # Clear buffers
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        
        # Start background reader thread
        self.running = True
        self.last_response = None
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()
    
    def _read_loop(self):
        """Background thread to continuously read from serial"""
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        with self.lock:
                            self.last_response = line
                        print(f"[RX] {line}")
            except:
                pass
            time.sleep(0.001)
    
    def _send_command(self, cmd):
        """Send command and wait for response"""
        with self.lock:
            self.last_response = None
        
        self.ser.write((cmd + '\n').encode())
        self.ser.flush()
        
        # Wait for response (up to 1 second)
        start_time = time.time()
        while time.time() - start_time < 1.0:
            with self.lock:
                if self.last_response is not None:
                    return self.last_response
            time.sleep(0.01)
        
        return None
    
    def get_state(self):
        """Read state from robot"""
        response = self._send_command('S')
        
        if response and response.startswith('STATE,'):
            try:
                values = [float(x) for x in response.split(',')[1:]]
                return np.array(values)
            except:
                print("ERROR: Failed to parse state")
                return None
        
        print("ERROR: No valid state response")
        return None
    
    def set_action(self, action):
        """Send action to robot"""
        action_str = 'A,' + ','.join([f"{x:.4f}" for x in action])
        response = self._send_command(action_str)
        
        return response == 'ACK'
    
    def reset(self):
        """Reset robot"""
        response = self._send_command('R')
        time.sleep(1.5)
        return response == 'RESET_DONE'
    
    def check_done(self):
        """Check termination"""
        response = self._send_command('T')
        
        if response and response.startswith('TERM,'):
            return bool(int(response.split(',')[1]))
        
        return False
    
    def close(self):
        """Close connection"""
        self.running = False
        time.sleep(0.1)
        self.ser.close()


# ========== DATA COLLECTOR (same as before) ==========
class DataCollector:
    def __init__(self, robot):
        self.robot = robot
        self.buffer = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': []
        }
    
    def collect_episode(self, policy, max_steps=500):
        """Collect one episode"""
        print("Resetting robot...")
        self.robot.reset()
        time.sleep(0.5)
        
        print("Reading initial state...")
        state = self.robot.get_state()
        
        if state is None:
            print("ERROR: Could not read initial state")
            return 0.0
        
        episode_reward = 0
        steps_completed = 0
        
        for step in range(max_steps):
            action = policy.get_action(state)
            
            success = self.robot.set_action(action)
            if not success:
                print(f"Failed to set action at step {step}")
                break
            
            time.sleep(0.02)  # 50 Hz
            
            next_state = self.robot.get_state()
            
            if next_state is None:
                print(f"Failed to read state at step {step}")
                break
            
            done = self.robot.check_done()
            reward = self.calculate_reward(state, action, next_state, done)
            
            self.buffer['states'].append(state.copy())
            self.buffer['actions'].append(action.copy())
            self.buffer['next_states'].append(next_state.copy())
            self.buffer['rewards'].append(reward)
            self.buffer['dones'].append(done)
            
            episode_reward += reward
            steps_completed += 1
            state = next_state
            
            if (step + 1) % 50 == 0:
                print(f"  Step {step+1}/{max_steps}, Reward: {episode_reward:.2f}")
            
            if done:
                print(f"Episode terminated at step {step+1}")
                break
        
        print(f"Episode: {steps_completed} steps, reward: {episode_reward:.2f}")
        return episode_reward
    
    def calculate_reward(self, state, action, next_state, done):
        """Reward function"""
        if next_state is None:
            return -10.0
        
        reward = 0.1  # Survival bonus
        
        if done:
            reward -= 5.0
        
        # Energy penalty
        reward -= np.sum(np.square(action)) * 0.001
        
        # Stability
        if len(next_state) >= 43:
            roll = next_state[36] * 90.0
            pitch = next_state[37] * 90.0
            reward -= (abs(roll) + abs(pitch)) * 0.01
        
        return reward
    
    def save_data(self, filename):
        """Save data"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'wb') as f:
            pickle.dump(self.buffer, f)
        
        print(f"\nData saved: {filename}")
        print(f"Transitions: {len(self.buffer['states'])}")


# ========== POLICIES ==========
class RandomPolicy:
    def __init__(self, action_dim=18):
        self.action_dim = action_dim
    
    def get_action(self, state):
        return np.random.uniform(-0.2, 0.2, self.action_dim)


class ScriptedPolicy:
    def __init__(self):
        self.t = 0
    
    def get_action(self, state):
        self.t += 1
        phase = (self.t * 0.02 * 1.0 * 2 * np.pi) % (2 * np.pi)
        
        action = np.zeros(18)
        for leg in range(6):
            offset = 0 if leg % 2 == 0 else np.pi
            action[leg * 3 + 0] = 0.15 * np.sin(phase + offset)
            action[leg * 3 + 1] = 0.2 * np.sin(phase + offset)
            action[leg * 3 + 2] = 0.15 * np.cos(phase + offset)
        
        return action


# ========== MAIN ==========
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    print("="*50)
    print("RL Spider Data Collection")
    print("="*50)
    
    port = input("Enter COM port (default COM3): ").strip() or "COM3"
    
    try:
        robot = RobotInterface(port=port)
        
        # Test communication
        print("\nTesting communication...")
        test_state = robot.get_state()
        if test_state is None:
            print("ERROR: Cannot communicate with robot!")
            exit(1)
        else:
            print(f"✓ Communication OK! State dimension: {len(test_state)}")
        
        collector = DataCollector(robot)
        
        policy = ScriptedPolicy()
        print("Using scripted tripod gait")
        
        num_episodes = int(input("Episodes (default 20): ") or "20")
        
        for episode in range(num_episodes):
            print(f"\n{'='*50}")
            print(f"Episode {episode+1}/{num_episodes}")
            print(f"{'='*50}")
            
            input("Place robot in start position, press Enter...")
            
            reward = collector.collect_episode(policy, max_steps=500)
            print(f"Episode reward: {reward:.2f}")
            
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\nStopped")
    
    finally:
        if 'collector' in locals():
            collector.save_data("data/iteration_0.pkl")
        if 'robot' in locals():
            robot.close()