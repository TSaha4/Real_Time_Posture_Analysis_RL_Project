# UPRYT - Real-Time Posture Analysis with Reinforcement Learning

An advanced posture correction system using RL for adaptive feedback with auto-switching algorithms, attention tracking, and hand tracking.

## Quick Start

### GUI Application (Recommended)
```bash
python gui_app.py
```

### Command Line
```bash
# Full system (posture + attention + hands)
python main.py --mode combined --algorithm ppo --audio

# Posture only
python main.py --mode realtime --algorithm ppo

# Train RL agent (recommended: use --enhanced-training)
python main.py --mode train --algorithm ppo --episodes 2000 --enhanced-training

# Train both PPO and DQN
python main.py --mode train-all --episodes 2000 --enhanced-training
```

## Features

### Core
- Real-time pose detection using MediaPipe (33 landmarks)
- Reinforcement Learning feedback (DQN, PPO, or Rule-Based)
- User calibration for personalized assessment
- Compact visual overlay (minimal screen coverage)
- **Auto-Switching**: Automatically selects best algorithm based on correction rate

### Tracking Modes
| Mode | Description |
|------|-------------|
| `combined` | Full system - posture + attention + hands |
| `realtime` | Real-time posture monitoring |
| `hand` | Hand/typing posture tracking only |
| `attention` | Face/gaze/focus tracking only |
| `train` | Train RL agent |
| `train-all` | Train both PPO and DQN |
| `compare` | Benchmark algorithms |

### Training
- **Enhanced Training**: Curriculum learning with domain randomization
- **LR Scheduling**: Learning rate decays during training for stability
- **Realistic Simulation**: User behavior modeling (fatigue, compliance, motivation)
- **Difficulty Levels**: easy, medium, hard

### Visualization
- **Training Visualization**: Generate plots from training results
- **Benchmark Metrics**: Correction rate, avg reward, total alerts

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Project Structure

```
UPRYT Project
├── main.py                    # CLI entry point (all modes)
├── gui_app.py                 # Tkinter GUI application
├── config.py                  # Centralized configuration
│
├── Core Modules
│   ├── pose_module.py         # MediaPipe integration
│   ├── posture_module.py      # Posture classification
│   ├── environment.py         # RL environment
│   ├── feedback.py            # Visual overlay
│   └── utils.py               # Utilities
│
├── RL Agents
│   ├── rl_agent.py           # DQN implementation
│   └── rl_ppo_agent.py       # PPO implementation
│
├── Training
│   └── simulation_enhanced.py # Enhanced training with curriculum
│
├── Tracking (Optional)
│   ├── attention_tracker.py   # Face/gaze detection
│   ├── hand_tracker.py        # Typing posture
│   └── combined_analyzer.py   # Unified analyzer
│
├── Utilities
│   ├── algorithm_selector.py  # Auto-switching algorithm
│   ├── audio_alerts.py        # Audio system
│   ├── visualize_results.py   # Training visualization
│   ├── user_profiles.py       # User profiles
│   └── online_learning.py     # Real-time learning
│
├── models/                    # Trained RL models (.pth)
└── results/                   # Training results & visualizations
```

## CLI Options

```bash
# Modes
--mode=combined    # Full system (recommended)
--mode=realtime    # Posture only
--mode=hand        # Hand tracking
--mode=attention   # Attention tracking
--mode=train       # Train RL agent
--mode=train-all   # Train both PPO and DQN
--mode=compare     # Benchmark algorithms

# Algorithms
--algorithm=ppo    # Proximal Policy Optimization
--algorithm=dqn    # Deep Q-Network
--algorithm=rule   # Rule-based (no RL)
--algorithm=auto   # Automatic algorithm selection (recommended)

# Feature Flags
--audio            # Enable audio alerts
--attention        # Enable attention tracking
--hands            # Enable hand tracking
--skip-calibration # Use default baseline
--enhanced-training # Use curriculum learning (recommended for training)
--auto-switch      # Enable automatic algorithm switching (default: on)
--no-auto-switch   # Disable automatic algorithm switching

# Training Options
--episodes=N       # Training episodes (default: 500)
--users=N          # Simulated users (default: 5)
--difficulty={easy,medium,hard} # Training difficulty

# Other
--camera=N         # Camera index (default: 0)
```

## Usage Examples

```bash
# Full featured session with auto-switching (recommended)
python main.py --mode combined --algorithm auto --audio

# Quick start (no calibration)
python main.py --mode realtime --algorithm rule --skip-calibration --audio

# Train PPO with enhanced training (recommended)
python main.py --mode train --algorithm ppo --episodes 2000 --enhanced-training --difficulty medium

# Train both PPO and DQN
python main.py --mode train-all --episodes 2000 --enhanced-training

# Visualize training results
python visualize_results.py --benchmark
```

## Training Visualization

After training, visualize results:

```bash
# Auto-detect latest training runs
python visualize_results.py

# With benchmark comparison
python visualize_results.py --benchmark

# Custom files
python visualize_results.py --ppo-file results/ppo_20260402_123456.json --dqn-file results/dqn_20260402_123456.json

# Adjust moving average window
python visualize_results.py --window 20
```

Generates:
- `results/ppo_training_viz.png` - PPO training curves
- `results/dqn_training_viz.png` - DQN training curves
- `results/comparison_viz.png` - PPO vs DQN comparison
- `results/benchmark_viz.png` - Benchmark metrics

## How It Works

1. **Camera** → MediaPipe Pose extracts 33 body landmarks
2. **Features** → Compute angles (neck, shoulders, spine)
3. **Classify** → Determine posture state (GOOD/SLOUCHING/etc)
4. **RL Agent** → Decide feedback action (no/subtle/strong)
5. **Feedback** → Visual overlay + audio alert

### State Space (18-dim)
```
[posture_label, score, consecutive_alerts, fatigue, motivation, frustration,
 is_good, badness, episode_alerts, correction_ratio, correction_streak, max_streak,
 compliance, stubbornness, attention_span, learning_rate, is_easy, is_hard]
```

### Action Space
```
0 = NO_FEEDBACK
1 = SUBTLE_ALERT
2 = STRONG_ALERT
```

### Rewards (Training)
- Successful correction: +1.5 to +2.3 (with time bonus)
- Alert ignored: -0.3
- Good posture maintained: +0.05 to +0.2

## Configuration

Edit `config.py` or use command-line flags:

```python
from config import config

# RL Settings
config.rl.batch_size = 128
config.rl.learning_rate = 0.001
config.rl.lr_decay = 0.5
config.rl.lr_decay_interval = 500

# System
config.system.decision_interval = 2.5  # seconds

# Posture thresholds
config.posture.neck_angle_threshold = 20.0
```

## Dependencies

- Python 3.8+
- OpenCV
- MediaPipe
- PyTorch
- NumPy
- Matplotlib (for visualization)

## License

MIT License