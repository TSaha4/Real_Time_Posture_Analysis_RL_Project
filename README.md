# UPRYT v2 - Real-Time Posture Analysis with Reinforcement Learning

![UPRYT Logo](logo/upryt_white.png)

An advanced posture correction system using RL for adaptive feedback with enhanced training, gamification, attention tracking, hand tracking, and mobile app support.

## Quick Start

### GUI Application (Recommended)
```bash
python gui_app.py
```

### Command Line
```bash
# Full system (posture + attention + hands + dashboard)
python main.py --mode=combined --algorithm=ppo --audio

# Posture only
python main.py --mode=realtime --algorithm=ppo

# Train RL agent
python main.py --mode=train --algorithm=ppo --episodes=500
```

## Features

### Core
- Real-time pose detection using MediaPipe (33 landmarks)
- Reinforcement Learning feedback (DQN, PPO, or Rule-Based)
- User calibration for personalized assessment
- Compact visual overlay (minimal screen coverage)

### Tracking Modes
| Mode | Description |
|------|-------------|
| `combined` | Full system - posture + attention + hands + dashboard |
| `realtime` | Real-time posture monitoring |
| `hand` | Hand/typing posture tracking only |
| `attention` | Face/gaze/focus tracking only |

### Training
- **Curriculum Learning**: Easy → Medium → Hard progression
- **Domain Randomization**: State noise for robustness
- **Online Learning**: Continue learning from real sessions

### Analytics
- **Attention Tracking**: Detects FOCUSED/DISTRACTED/AWAY states
- **Hand Tracking**: Monitors typing posture (GOOD/TENSE/ASYMMETRIC)
- **Gamification**: Achievements, streaks, weekly trends
- **Dashboard Export**: Session data for web dashboards

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Project Structure

```
├── main.py                    # CLI entry point (all modes)
├── gui_app.py                 # Tkinter GUI application
├── config.py                  # Centralized configuration
│
├── Core Modules
│   ├── pose_module.py        # MediaPipe integration
│   ├── posture_module.py      # Posture classification
│   ├── environment.py        # RL environment
│   ├── feedback.py           # Visual overlay (compact)
│   └── audio_alerts.py      # Audio system
│
├── RL Agents
│   ├── rl_agent.py          # DQN implementation
│   └── rl_ppo_agent.py      # PPO implementation
│
├── Training
│   ├── simulation.py         # Basic training
│   └── simulation_enhanced.py # Curriculum + domain randomization
│
├── Tracking
│   ├── attention_tracker.py   # Face/gaze detection
│   ├── hand_tracker.py       # Typing posture
│   └── combined_analyzer.py  # Unified analyzer
│
├── User Management
│   ├── user_profiles.py     # Profiles & gamification
│   ├── online_learning.py   # Real-time learning
│   └── web_dashboard.py     # Data export
│
├── Mobile App
│   └── mobile_app/          # Android APK builder (Kivy)
│
├── data/
│   ├── profiles/            # User profiles
│   └── dashboard/            # Session data
├── models/                   # Trained RL models
└── logs/                    # Session logs
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
--algorithm=rule    # Rule-based (no RL)

# Feature Flags
--audio            # Enable audio alerts
--attention        # Enable attention tracking
--hands            # Enable hand tracking
--dashboard        # Enable dashboard export
--online-learning  # Learn from sessions
--skip-calibration # Use default baseline
--enhanced-training # Use curriculum learning

# Other
--camera=N         # Camera index (default: 0)
--episodes=N       # Training episodes (default: 500)
--users=N         # Simulated users (default: 5)
```

## Usage Examples

```bash
# Full featured session
python main.py --mode=combined --algorithm=ppo --audio

# Quick start (no calibration)
python main.py --mode=realtime --algorithm=rule --skip-calibration --audio

# Train with enhanced simulator
python main.py --mode=train --algorithm=ppo --episodes=1000 --enhanced-training

# Compare algorithms
python main.py --mode=compare
```

## Mobile App

Build an Android APK from the project:

```bash
cd mobile_app

# Using Docker (recommended)
docker build -t upryt-builder .
docker run -v $(pwd):/app upryt-builder

# Or on Linux
pip install buildozer
./build.sh
```

See `mobile_app/README.md` for full instructions.

## How It Works

1. **Camera** → MediaPipe Pose extracts 33 body landmarks
2. **Features** → Compute angles (neck, shoulders, spine)
3. **Classify** → Determine posture state (GOOD/SLOUCHING/etc)
4. **RL Agent** → Decide feedback action (no/sublte/strong)
5. **Feedback** → Visual overlay + audio alert

### State Space (6-dim)
```
[posture_label, score, duration_bad, time_since_alert, corrections, consecutive_alerts]
```

### Action Space
```
0 = NO_FEEDBACK
1 = SUBTLE_ALERT
2 = STRONG_ALERT
```

## Configuration

Edit `config.py` or use command-line flags:

```python
from config import config

# Training
config.training.enable_curriculum = True
config.training.enable_domain_randomization = True

# System
config.system.decision_interval = 2.5  # seconds

# Posture thresholds
config.posture.neck_angle_threshold = 20.0
```

## Dependencies

- Python 3.8+
- OpenCV 4.8.1
- MediaPipe 0.10.7
- PyTorch 2.1.0
- NumPy 1.24.3
- Pillow 10.1.0

## License

MIT License
