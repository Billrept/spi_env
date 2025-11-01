# Hexapod Balance Control - RL Training System

This package contains the reinforcement learning (RL) training system for hexapod balance control using PPO (Proximal Policy Optimization).

## Components

### 1. `hexapod_env.py` - Gymnasium Environment
Custom environment implementing:
- **Observation space**: 15D (IMU data + 6 joint positions)
  - Roll, pitch, yaw (rad)
  - Gyroscope x, y, z (rad/s)
  - Accelerometer x, y, z (g)
  - 6 joint positions (rad, relative to neutral)
  
- **Action space**: 6D position deltas (rad per step)
  - Clamped to ±0.05 rad by default
  - Exponential smoothing applied
  
- **Reward function**:
  - Uprightness: -(roll² + pitch²)
  - Smoothness: -λ₁ × ||Δaction||²
  - Joint velocity penalty: -λ₂ × ||joint_vel||²
  - Alive bonus per step
  - Optional stepping bonus (phase 2)
  
- **Features**:
  - Domain randomization (IMU bias, action noise, timing jitter)
  - Configurable fall thresholds
  - Simplified physics simulation
  - Conversion utilities (ticks ↔ radians, hundredths ↔ radians)

### 2. `train_ppo.py` - Training Script
PPO training with Stable-Baselines3:
- **Hyperparameters**:
  - 8 parallel environments
  - 1024 steps per update
  - Learning rate: 3e-4 (linear schedule)
  - Gamma: 0.99, GAE lambda: 0.95
  - Clip range: 0.2
  - Entropy coefficient: 0.005
  
- **Curriculum learning**:
  - Phase 1: Balance-only (uprightness reward)
  - Phase 2: Stepping enabled when avg reward > threshold
  
- **Callbacks**:
  - Evaluation with best model saving
  - Regular checkpointing
  - TensorBoard logging

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install gymnasium numpy stable-baselines3 torch tensorboard
```

## Usage

### Training

```bash
# Basic training (2M timesteps, curriculum learning)
python train_ppo.py

# Custom training
python train_ppo.py \
    --timesteps 5000000 \
    --n-envs 16 \
    --learning-rate 1e-4 \
    --curriculum-threshold 8.0 \
    --output-dir ./my_runs \
    --seed 123

# Training without curriculum
python train_ppo.py --no-curriculum
```

**Training outputs**:
- `runs/ppo_hexapod_YYYYMMDD_HHMMSS/`
  - `best_model/` - Best performing model
  - `checkpoints/` - Periodic checkpoints
  - `final_model.zip` - Final trained model
  - `tensorboard/` - TensorBoard logs
  - `eval/` - Evaluation metrics
  - `progress.csv` - Training progress

### Evaluation

```bash
# Evaluate a trained model
python train_ppo.py --eval runs/ppo_hexapod_YYYYMMDD_HHMMSS/best_model.zip --eval-episodes 20
```

### Monitoring Training

```bash
# Start TensorBoard
tensorboard --logdir runs/

# Open browser to http://localhost:6006
```

## Configuration

### Environment Parameters

Edit `hexapod_env.py` constants:

```python
# Safety limits (adjust based on physical robot)
JOINT_LIMITS_RAD = np.array([
    [-np.pi/2, np.pi/2],   # Joint 1
    # ... adjust for your robot
])

# Reward weights (tune for performance)
REWARD_WEIGHTS = {
    "upright": 1.0,      # Uprightness importance
    "smooth": 0.01,      # Action smoothness
    "joint_vel": 0.005,  # Joint velocity penalty
    "alive": 0.5,        # Alive bonus
    "step_bonus": 0.05,  # Stepping reward (phase 2)
}
```

### Training Parameters

Modify `train_ppo.py` or use command-line args:

```python
# Training configuration
n_envs = 8              # More envs = faster training, more CPU
n_steps = 1024          # Steps per update
batch_size = 1024       # Minibatch size
learning_rate = 3e-4    # Higher = faster learning, less stable

# Curriculum learning
curriculum_threshold = 5.0  # Reward threshold for phase 2
```

## Expected Training Progress

**Phase 1 (Balance only)**:
- Steps 0-500k: Agent learns to keep roll/pitch near zero
- Steps 500k-1M: Reward should increase from ~-2 to ~5
- Policy learns to counteract disturbances

**Phase 2 (Stepping - if enabled)**:
- Steps 1M-2M: Agent attempts leg coordination
- Stepping bonus encourages alternating motion
- Balance maintained while moving

**Convergence**: ~2M steps for basic balance, 5M+ for robust stepping

## Troubleshooting

### Training unstable
- Reduce `learning_rate` (try 1e-4)
- Increase `action_delta_clamp` limit
- Adjust `action_smoothing_alpha` (lower = more smoothing)
- Check `fall_threshold_deg` (too tight?)

### Reward not improving
- Check observation normalization
- Verify physics simulation makes sense
- Tune reward weights (increase `upright` weight)
- Reduce domain randomization initially

### Agent too conservative
- Increase `ent_coef` (exploration)
- Reduce smoothness penalties
- Check action space isn't over-constrained

## Integration with Hardware

This RL system trains a policy in simulation. To deploy on real hardware:

1. **Train policy** using this code
2. **Export model**: Model saved as `.zip` file
3. **Load in live runner**: Use PC-side runner with UART transport
4. **Connect to CM-550**: Stream observations, send actions at 50Hz

See other components for UART protocol and CM-550 bridge implementation.

## Notes

- Physics simulation is simplified - replace with proper dynamics for better sim-to-real transfer
- Domain randomization helps bridge sim-to-real gap
- Start with conservative parameters, then tune based on real robot behavior
- Monitor IMU data carefully - ensure proper calibration
- Test extensively in simulation before deploying to hardware

## Dependencies

- Python 3.8+
- gymnasium
- numpy
- stable-baselines3
- torch (CPU or CUDA)
- tensorboard (optional, for monitoring)

## License

See repository LICENSE file.
