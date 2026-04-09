# UPRYT - Real-Time Posture Analysis with Reinforcement Learning

An enterprise-grade posture correction system using reinforcement learning for adaptive feedback. Designed for production deployment in corporate wellness, remote work, and ergonomic applications.

## Quick Start

### GUI Application (Recommended)
```bash
python gui_app.py
```

### Command Line
```bash
# Run realtime with PPO (recommended RL agent)
python main.py --mode=realtime --algorithm=ppo

# Run realtime with DQN
python main.py --mode=realtime --algorithm=dqn

# Run with rule-based (fallback)
python main.py --mode=realtime --algorithm=rule

# Train both algorithms
python main.py --mode=train-all --episodes=150 --users=5
```

## Features

### Core
- Real-time pose detection using MediaPipe (33 body landmarks)
- Reinforcement Learning feedback (PPO, DQN, or Rule-Based)
- User calibration for personalized assessment
- Compact visual overlay
- **Auto-Switching**: Automatically selects best algorithm based on correction rate

### Posture Detection
- Forward head posture
- Slouching/spine curvature
- Leaning/shoulder asymmetry
- Good posture recognition

### Training
- **Enhanced Training**: Simulated user behavior modeling
- **Difficulty Levels**: easy, medium, hard
- **Curriculum Learning**: Progressive difficulty stages
- **Domain Randomization**: Robust training

## Project Structure

```
UPRYT Project
├── main.py                    # CLI entry point (all modes)
├── gui_app.py                 # Tkinter GUI application
├── config.py                  # Centralized configuration
│
├── Core Modules
│   ├── pose_module.py         # MediaPipe pose detection
│   ├── posture_module.py    # Posture classification
│   ├── environment.py      # RL environment & state
│   ├── feedback.py          # Visual overlay
│   └── utils.py             # Logging utilities
│
├── RL Agents
│   ├── rl_agent.py         # DQN implementation
│   └── rl_ppo_agent.py     # PPO implementation
│
├── Training
│   └── simulation_enhanced.py # Training simulator
│
├── Tracking (Optional)
│   ├── attention_tracker.py   # Face/gaze detection
│   ├── hand_tracker.py        # Typing posture
│   └── combined_analyzer.py   # Unified analyzer
│
├── Utilities
│   ├── algorithm_selector.py  # Auto-switching
│   ├── audio_alerts.py        # Audio alerts
│   ├── visualize_results.py   # Training visualization
│   ├── user_profiles.py     # User profiles
│   └── online_learning.py   # Real-time learning
│
├── models/                  # Trained RL models (.pth)
└── results/                # Training results
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## CLI Options

### Modes
| Mode | Description |
|------|-------------|
| `realtime` | Real-time posture monitoring |
| `train` | Train RL agent |
| `train-all` | Train both PPO and DQN |
| `compare` | Benchmark algorithms |
| `hand` | Hand/typing posture only |
| `attention` | Face/gaze tracking only |
| `combined` | Full system |

### Algorithms
| Algorithm | Description |
|-----------|-------------|
| `ppo` | Proximal Policy Optimization (recommended) |
| `dqn` | Deep Q-Network |
| `rule` | Rule-based (fallback) |
| `auto` | Automatic selection |

### Flags
| Flag | Description |
|------|-------------|
| `--audio` | Enable audio alerts |
| `--skip-calibration` | Use default baseline |
| `--enhanced-training` | Use curriculum learning |
| `--attention` | Enable attention tracking |
| `--hands` | Enable hand tracking |
| `--dashboard` | Enable session export |

### Training Options
| Option | Default | Description |
|--------|---------|-------------|
| `--episodes` | 500 | Training episodes |
| `--users` | 5 | Simulated users |
| `--difficulty` | medium | easy/medium/hard |

## Usage Examples

```bash
# Recommended: PPO with calibration (in production)
python main.py --mode=realtime --algorithm=ppo

# Quick start: Use default calibration
python main.py --mode=realtime --algorithm=ppo --skip-calibration

# Rule-based (for comparison or debugging)
python main.py --mode=realtime --algorithm=rule

# Train PPO
python main.py --mode=train --algorithm=ppo --episodes=150 --users=5

# Train both PPO and DQN (recommended)
python main.py --mode=train-all --episodes=150 --users=5

# Benchmark comparison
python main.py --mode=compare --episodes=50
```

## Training Visualization

```bash
# Generate plots from training
python visualize_results.py

# Benchmark comparison
python visualize_results.py --benchmark
```

Generates:
- `results/ppo_training_viz.png` - PPO training curves
- `results/dqn_training_viz.png` - DQN training curves
- `results/comparison_viz.png` - PPO vs DQN comparison
- `results/benchmark_viz.png` - Benchmark metrics

## How It Works

### Pipeline

1. **Camera Capture** → MediaPipe Pose extracts 33 body landmarks
2. **Feature Extraction** → Compute geometric features (neck angle, shoulder diff, spine inclination)
3. **Calibration** → Establish user-specific baseline (20 frames)
4. **Classification** → Determine posture state (GOOD/SLOUCHING/FORWARD_HEAD/LEANING)
5. **RL Decision** → PPO/DQN agent selects action (NO_FEEDBACK/SUBTLE_ALERT/STRONG_ALERT)
6. **Feedback** → Visual overlay + optional audio alert

### State Space (18-dim)
```
[posture_label, score, consecutive_alerts, fatigue, motivation, frustration,
 is_good, badness, episode_alerts, correction_ratio, correction_streak, max_streak,
 compliance, attention_span, stubbornness, session_time, correction_ability, alert_sensitivity]
```

### Action Space
| Action | Value | Description |
|--------|-------|-------------|
| NO_FEEDBACK | 0 | No alert - good posture or ignored |
| SUBTLE_ALERT | 1 | Gentle notification |
| STRONG_ALERT | 2 | Urgent alert |

### Reward Function
| Scenario | Reward |
|----------|--------|
| Successful correction | +1.5 to +2.3 |
| Alert ignored | -0.3 |
| Good posture maintained | +0.05 |
| Sustained bad posture | -0.1 |

## Configuration

Edit `config.py`:

```python
from config import config

# RL Settings
config.rl.state_size = 18
config.rl.action_size = 3
config.rl.batch_size = 128
config.rl.learning_rate = 0.001

# System Settings
config.system.decision_interval = 2.5  # seconds between decisions

# Posture Thresholds
config.posture.neck_angle_threshold = 15.0
config.posture.shoulder_diff_threshold = 12.0
config.posture.spine_inclination_threshold = 18.0
config.posture.forward_head_threshold = 15.0
```

### Key Parameters

| Parameter | Value | Description |
|----------|-------|-------------|
| `decision_interval` | 2.5s | Time between RL decisions |
| `calibration_frames` | 20 | Frames for calibration |
| `cooldown_period` | 3.0s | Minimum between alerts |
| `neck_angle_threshold` | 15.0 | Max deviation from baseline |
| `shoulder_diff_threshold` | 12.0 | Max shoulder asymmetry |

## Dependencies

- Python 3.8+
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- PyTorch
- NumPy
- Matplotlib (optional, for visualization)

## Production Deployment

### Recommended Setup
```bash
# Train once before deployment
python main.py --mode=train-all --episodes=200 --users=8

# Start realtime session
python main.py --mode=realtime --algorithm=ppo
```

### Monitoring
- Session logs saved to: `logs/`
- Trained models in: `models/`
- Results in: `results/`

## Performance

| Metric | PPO | DQN | Rule-Based |
|--------|-----|-----|-----------|
| Avg Reward | 20-30 | 15-25 | N/A |
| Correction Rate | 12-15% | 12-15% | 8-12% |
| Alert Frequency | Moderate | Moderate | High |

## License

MIT License