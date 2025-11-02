# SPI-RL (12-Joint) — PPO Training & Live Control via Serial IMU

Balance a 6-leg “SPI” robot with **12 virtual joints** (6 Horizontal + 6 Vertical) using **Stable-Baselines3 (SB3) PPO**.  
Train **offline** (no hardware required) with a lightweight toy dynamics, then deploy **online** by streaming actions to your CM-550/board over **serial** while parsing the board’s **text IMU** output.

---

## Features

- **Gymnasium** environment with **12 actions** `[H1..H6, V1..V6]` and IMU observations.
- **Two modes**:
  - **Offline** (default) — mock dynamics; fast PPO training on your computer.
  - **Online** — connect to your board: read IMU text and send joint targets via Remocon packets.
- **SB3 PPO** training with optional **VecNormalize** support.
- Clean **serial protocol**: `FF 55 LB ~LB HB ~HB` with **4-bit channel** + **12-bit ticks**.
- Safety: per-step clamps & joint limits to avoid sudden moves.

---

## Project Layout

```
spi_env/
├─ remocon12_link.py        # Serial I/O (online mode): parse IMU text, send 12 joint targets
├─ spi_balance12_env.py     # Gym env (offline/online switch; 12 delta actions)
├─ train_spi.py             # PPO trainer (offline by default)
├─ run_policy.py            # Run a saved policy online (opens serial)
└─ README.md                # This file
```

---

## Requirements

- Python **3.10+** (tested with 3.13)
- Packages:
  - `gymnasium>=1.0.0`
  - `stable-baselines3>=2.3.0`
  - `torch`
  - `numpy`
  - `pyserial`
  - `tensorboard` (optional for logs)

### Setup

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
. .venv/Scripts/activate

pip install --upgrade pip
pip install "gymnasium>=1.0.0" "stable-baselines3>=2.3.0" torch numpy pyserial tensorboard
```

> In VS Code, set the interpreter to your `.venv` (Command Palette → “Python: Select Interpreter”).

---

## Quickstart

### 1) Train **offline** (no hardware)
```bash
python train_spi.py
```
- Creates an **offline** env (`use_hardware=False`).
- Trains PPO to minimize roll/pitch using a toy dynamics model.
- Saves model to `./runs/final/ppo_spi12_offline.zip`.

### 2) Run policy **online** (with your robot)
1) Ensure your board prints IMU lines similar to:
```
[RX] RAW  R/P/Y(deg): 0.39 -1.54 14.24 | ACC(g): 0.029 0.006 0.967 | GYRO(dps): -0.05 0.01
[RX] KF   R/P(deg):   0.43 -1.58
```
(KF line optional; if present, it overrides roll/pitch.)

2) Edit serial settings in **`remocon12_link.py`**:
```python
PORT = "COM3"     # or "/dev/tty.usbserial-XXXX" on macOS/Linux
BAUD = 57600
```

3) Run:
```bash
python run_policy.py
```
Edit `SERIAL_PORT` and `MODEL_PATH` inside `run_policy.py` if needed.

---

## Configuration (important)

Open **`remocon12_link.py`**:

- **Joint limits** (radians):
```python
JOINT_LIMITS_H = [(-2.094, 2.094)] * 6  # Horizontal (H1..H6)
JOINT_LIMITS_V = [(-2.094, 2.094)] * 6  # Vertical   (V1..V6)
```

- **Channel mapping** (4-bit channel IDs 0..11):
```python
CH_MAP_H = [0, 1, 2, 3, 4, 5]       # H1..H6 → channels 0..5
CH_MAP_V = [6, 7, 8, 9, 10, 11]     # V1..V6 → channels 6..11
```

Open **`spi_balance12_env.py`**:

- **Per-step clamps** (keep movements small, esp. at first):
```python
MAX_STEP_RAD_H = 0.05
MAX_STEP_RAD_V = 0.05
```

- **Episode length and cadence**:
```python
CTRL_HZ = 50               # control rate
EPISODE_SECONDS = 10.0     # → 500 steps/episode
FALL_DEG = 45.0            # terminate if |roll| or |pitch| exceeds this
```

---

## How It Works

### Observations (21-D)
```
[roll, pitch, yaw, gx, gy, gz, ax, ay, az, qH1..qH6, qV1..qV6]

```
- IMU: angles in **radians**, gyro in **rad/s**, accel in **g**.
- `qH*`, `qV*`: **commanded** angles (what we send) — used to keep state Markovian when we don’t have motor feedback.

### Actions (12-D)
Per-step **delta angles** (radians):
```
[ ΔH1..ΔH6, ΔV1..ΔV6 ]
```
- Each delta is clamped per step (`MAX_STEP_RAD_*`).
- The integrated commanded angles are clamped to **hard joint limits**.

### Reward
Dense and smooth:
```
reward = − (roll^2 + pitch^2)  −  1e−3 * ||Δaction||^2
```

---

## Serial Protocol (Remocon)

- **Packet**: `FF 55 LB ~LB HB ~HB`
- **16-bit value encoding**:
  - Upper **4 bits**: `channel` (0..11)
  - Lower **12 bits**: `ticks` (0..4095)
- **Conversion**: `ticks = round(radians * 4096 / (2π))` (clamped)
- On the board: `chan = value >> 12`, `ticks = value & 0x0FFF`, then route to the corresponding joint.

> At 57,600 bps, 12 packets @ 50 Hz is tight. If you see lag, reduce `CTRL_HZ` (e.g., 33 Hz) or raise `BAUD` on both sides.

---

## Training Variants

### A) Default (no normalization)
`train_spi.py` (as provided) trains without VecNormalize:
```python
from gymnasium.wrappers import RecordEpisodeStatistics
env = RecordEpisodeStatistics(SPIBalance12Env(use_hardware=False))
model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=128, ...)
```

### B) With VecNormalize (recommended once stable)

**Train:**
```python
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RecordEpisodeStatistics

def env_fn():
    return RecordEpisodeStatistics(SPIBalance12Env(use_hardware=False))

train_venv = DummyVecEnv([env_fn])
train_venv = VecNormalize(train_venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
model = PPO("MlpPolicy", train_venv, n_steps=1024, batch_size=256, ...)
model.learn(total_timesteps=300_000)
train_venv.save("./runs/final/vecnorm.pkl")
model.save("./runs/final/ppo_spi12_offline")
```

**Run online with saved stats:**
```python
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_online_env():
    return SPIBalance12Env(use_hardware=True, port="COM3")

base = DummyVecEnv([make_online_env])
venv = VecNormalize.load("./runs/final/vecnorm.pkl", base)
venv.training = False
venv.norm_reward = False

model = PPO.load("./runs/final/ppo_spi12_offline.zip", env=venv)
```

---

## Troubleshooting

- **`RecordEpisodeStatistics` import error**  
  Use Gymnasium wrapper:  
  `from gymnasium.wrappers import RecordEpisodeStatistics`

- **Gymnasium seeding / Pylance warnings**  
  The env uses a private NumPy RNG; no `gymnasium.utils.seeding` calls.

- **VecNormalize assertion**  
  Train **and** eval must be wrapped identically (`DummyVecEnv` → `VecNormalize`), and eval must load/sync stats.

- **No IMU data online**  
  Check `PORT`/`BAUD`, cable, and that the board prints `"RAW"` lines (and optionally `"KF"`).

- **Robot jumps on start**  
  Keep `MAX_STEP_RAD_*` small (0.02–0.05). Start with all commanded angles at 0.0 (neutral).

---

## Safety

- Verify **joint limits** before live runs.
- Start at **low control rates** (33–50 Hz) and small per-step deltas.
- Ensure mechanical clearances; keep a kill-switch (unplug/disable torque) within reach.

---

## Extending

- **Real joint feedback:** print ticks `p1..p12` from the board and parse them to replace commanded angles in the observation.
- **Better offline model:** swap the toy dynamics for a simple simulator or a learned dynamics model.
- **Absolute-action mode:** change from delta to absolute commands (keep a per-step clamp to prevent jerks).

---

## License

Choose what fits your needs (e.g., MIT):

```
MIT License
Copyright (c) 2025 ...
Permission is hereby granted, free of charge, to any person obtaining a copy...
```

---

## Acknowledgements

Built with **Gymnasium** and **Stable-Baselines3**.  
Thanks to the SPI/CM-550 community for serial formats and IMU parsing patterns.

---

**Questions or edits?** Open an issue or ping in chat with mapping, units, or routines you want baked in.
