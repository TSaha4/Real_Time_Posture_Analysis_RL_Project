# UPRYT v2 - Usage Guide

Complete guide to using all features in UPRYT v2.

## Quick Reference

| Command | Description |
|---------|-------------|
| `python gui_app.py` | GUI application |
| `python main.py --mode=realtime --algorithm=ppo` | Real-time posture monitoring (recommended) |
| `python main.py --mode=realtime --algorithm=auto` | Posture with auto-switching |
| `python main.py --mode=train --algorithm=ppo --episodes=500` | Train PPO |
| `python main.py --mode=train --algorithm=dqn --episodes=500` | Train DQN |
| `python main.py --mode=train-all --episodes=500` | Train both (recommended) |
| `python main.py --mode=compare` | Benchmark algorithms |
| `python visualize_results.py` | Visualize training results |

## Project Structure

```
UPRYT/
├── Core Files (Required)
│   ├── main.py                 # CLI entry point
│   ├── config.py              # Configuration
│   ├── environment.py        # RL environment
│   ├── posture_module.py     # Posture classification
│   ├── pose_module.py       # MediaPipe pose detection
│   ├── feedback.py          # Visual overlay
│   ├── utils.py             # Logging utilities
│   ├── algorithm_selector.py # Auto-switching
│   ├── rl_agent.py          # DQN agent
│   ├── rl_ppo_agent.py      # PPO agent
│   └── simulation_enhanced.py # Training simulator
│
├── Optional Modules
│   ├── user_profiles.py     # User profiles & gamification
│   ├── online_learning.py   # Real-time learning updates
│   ├── audio_alerts.py      # Audio feedback
│   ├── hand_tracker.py      # Hand/typing posture tracking
│   ├── attention_tracker.py # Face/gaze tracking
│   ├── combined_analyzer.py # Unified analyzer
│   ├── web_dashboard.py     # Session dashboard export
│   ├── visualize_results.py # Training visualization
│   ├── gui_app.py           # GUI application
│   └── setup.py             # Package setup
│
├── models/                  # Trained RL models
│   ├── ppo_final.pth, ppo_best.pth
│   └── dqn_final.pth, dqn_best.pth
│
└── results/                # Training results & logs
```

## Modes

### Real-Time Posture (Recommended for Production)
```bash
# With PPO (recommended)
python main.py --mode=realtime --algorithm=ppo

# With auto-switching (automatic algorithm selection)
python main.py --mode=realtime --algorithm=auto

# With DQN
python main.py --mode=realtime --algorithm=dqn

# With rule-based (fallback)
python main.py --mode=realtime --algorithm=rule --skip-calibration
```

### Combined Mode
Full system: posture + attention + hands with automatic algorithm switching
```bash
python main.py --mode=combined --algorithm=auto --audio

# With specific algorithm
python main.py --mode=combined --algorithm=ppo --audio
```

### Standalone Trackers
```bash
python main.py --mode=hand        # Hand/typing posture
python main.py --mode=attention   # Face/gaze tracking
```

### Training
```bash
# Train PPO (recommended)
python main.py --mode=train --algorithm=ppo --episodes=500 --users=5 --difficulty=medium

# Train DQN
python main.py --mode=train --algorithm=dqn --episodes=500 --users=5 --difficulty=medium

# Train both PPO and DQN (recommended)
python main.py --mode=train-all --episodes=500 --users=5 --difficulty=medium

# Benchmark comparison
python main.py --mode=compare --episodes=50
```

### Visualization
```bash
# Auto-detect latest training runs
python visualize_results.py

# With benchmark metrics
python visualize_results.py --benchmark

# Custom files
python visualize_results.py --ppo-file results/ppo_20260402_123456.json --dqn-file results/dqn_20260402_123456.json
```

## All CLI Options

```
--mode={realtime,hand,attention,combined,train,train-all,compare}
--algorithm={ppo,dqn,rule,auto}    # auto = automatic algorithm selection

--camera=N           Camera index (default: 0)
--episodes=N        Training episodes (default: 500)
--users=N           Simulated users for training (default: 5)
--difficulty={easy,medium,hard}  Training difficulty (default: medium)

Feature Flags:
--audio             Enable audio alerts
--attention         Enable attention tracking
--hands             Enable hand tracking
--dashboard         Enable dashboard export
--skip-calibration  Skip calibration (use defaults)
--enhanced-training Use curriculum + domain randomization (recommended)
--auto-switch       Enable automatic algorithm switching (default: on)
--no-auto-switch    Disable automatic algorithm switching
--online-learning   Enable online learning
--no-audio          Disable audio alerts
--multi-camera      Enable multi-camera support
```

## GUI Application

Run `python gui_app.py` for:
1. **Mode Selection**: Combined, Realtime, Hand, Attention, Train, Compare
2. **Real-Time Options**: Algorithm, camera, audio, calibration
3. **Training Options**: Algorithm, episodes, users, enhanced training, difficulty
4. **Additional Features**: Attention, hands tracking

## Training Details

### Enhanced Training
The enhanced training system includes:
- **Curriculum Learning**: Progresses through difficulty levels
- **Domain Randomization**: Adds noise to states for robustness
- **Realistic User Simulation**: Models fatigue, compliance, motivation
- **LR Scheduling**: Learning rate decays every 500 episodes

### Difficulty Levels
- **easy**: High compliance (80-95%), low stubbornness, good attention span
- **medium**: Moderate compliance (50-75%), some stubbornness
- **hard**: Low compliance (20-50%), high stubbornness, poor attention span

### Reward Shaping
- Successful correction: +1.5 to +2.3 (with time bonus for quick corrections)
- Alert ignored: -0.3
- Good posture maintained with alert: +0.2
- Good posture maintained (no alert): +0.05
- Bad posture ignored: -0.1

### State Space (18-dim)
```
[posture_label, score, consecutive_alerts, fatigue, motivation, frustration,
 is_good, badness, episode_alerts, correction_ratio, correction_streak, max_streak,
 compliance, attention_span, stubbornness, session_time, correction_ability, alert_sensitivity]
```

### Action Space (3 actions)
| Action | Description |
|--------|-------------|
| 0 (NO_FEEDBACK) | No alert - good posture or ignored |
| 1 (SUBTLE_ALERT) | Gentle notification |
| 2 (STRONG_ALERT) | Urgent alert |

## Feature Details

### Attention Tracking
- **FOCUSED**: Looking at screen
- **DISTRACTED**: Looking away occasionally  
- **AWAY**: No face detected

### Hand Tracking
- **GOOD**: Balanced hand position
- **TENSE**: High tension
- **ASYMMETRIC**: Uneven hands
- **RAISED_ARMS**: Arms too high

### Auto-Switching Algorithm
Automatically selects the best algorithm based on:
- **Correction Rate**: % of alerts that lead to posture improvement
- **Average Posture Score**: Overall posture quality
- Switches every 30 seconds if another algorithm performs 10% better

### Posture Classification
- **GOOD**: Overall score >= 0.70
- **SLOUCHING**: Spine inclination > threshold
- **FORWARD_HEAD**: Head position forward of shoulders
- **LEANING**: Shoulder asymmetry detected

## Training Visualization

The visualization tool generates:

| File | Description |
|------|-------------|
| `ppo_training_viz.png` | Episode rewards, eval rewards, distribution, stats |
| `dqn_training_viz.png` | Same as PPO for DQN |
| `comparison_viz.png` | PPO vs DQN comparison |

### Benchmark Metrics
- **Correction Rate**: % of alerts that result in posture correction
- **Avg Episode Reward**: Mean reward per episode
- **Total Alerts**: Number of alerts sent during benchmark

## Python API Examples

### Training
```python
from simulation_enhanced import train_ppo, train_dqn
import random

random.seed(42)

# Train PPO (recommended)
stats = train_ppo(num_episodes=500, num_users=5, verbose=True, difficulty='medium')

# Train DQN
stats = train_dqn(num_episodes=500, num_users=5, verbose=True, difficulty='medium')

# Compare algorithms
from simulation_enhanced import compare_algorithms
results = compare_algorithms(
    ppo_path='models/ppo_final.pth',
    dqn_path='models/dqn_final.pth',
    num_episodes=50
)
print(results)
```

### Loading Trained Models
```python
from rl_agent import DQNAgent
from rl_ppo_agent import PPOPPOAgent

# Load PPO
ppo = PPOPPOAgent(state_size=18, action_size=3)
ppo.load('models/ppo_final.pth')

# Load DQN
dqn = DQNAgent(state_size=18, action_size=3)
dqn.load('models/dqn_final.pth')
```

### Using Posture Classifier
```python
from posture_module import RuleBasedClassifier, PostureLabel

classifier = RuleBasedClassifier()
classifier.set_baseline({
    'neck_angle': 45,
    'shoulder_diff': 5,
    'spine_inclination': 25,
    'forward_head_y': 20
})

label, score = classifier.classify({
    'neck_angle': 50,
    'shoulder_diff': 8,
    'spine_inclination': 30,
    'forward_head_y': 25
})
print(f"Posture: {label.value}, Score: {score:.2f}")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not working | Try `--camera=1` |
| Low video quality | Use 1080p webcam |
| No trained models | Run `--mode=train-all --episodes=500` |
| Laggy display | Disable unused tracking features (--attention, --hands) |
| Training unstable | Try `--difficulty easy` first |
| Poor correction rate | Train for more episodes (500+), try hard difficulty |
| Import errors | Ensure all dependencies installed: `pip install -r requirements.txt` |

## Configuration

Edit `config.py` for custom settings:

```python
from config import config

# RL Configuration
config.rl.batch_size = 128
config.rl.learning_rate = 0.001
config.rl.gamma = 0.99
config.rl.lr_decay = 0.5
config.rl.lr_decay_interval = 500

# System Settings
config.system.decision_interval = 2.5    # seconds between RL decisions
config.system.calibration_frames = 20    # frames for baseline
config.system.cooldown_period = 3.0      # minimum between alerts

# Posture Thresholds
config.posture.neck_angle_threshold = 15.0
config.posture.shoulder_diff_threshold = 12.0
config.posture.spine_inclination_threshold = 18.0
config.posture.forward_head_threshold = 15.0
```

## Performance Metrics (Expected)

| Metric | PPO | DQN | Rule-Based |
|--------|-----|-----|-----------|
| Avg Reward | 20-30 | 15-25 | N/A |
| Correction Rate | 12-15% | 12-15% | 8-12% |
| Alert Frequency | Moderate | Moderate | High |

## Files Generated

### Training
- `results/ppo_{timestamp}.json` - PPO training metrics
- `results/dqn_{timestamp}.json` - DQN training metrics
- `models/ppo_final.pth` - Final PPO model
- `models/ppo_best.pth` - Best PPO model (highest eval reward)
- `models/dqn_final.pth` - Final DQN model
- `models/dqn_best.pth` - Best DQN model

### Session Logs
- `logs/training_metrics.jsonl` - Real-time training metrics
- `logs/session_{timestamp}.json` - Session data

### Visualization
- `results/ppo_training_viz.png`
- `results/dqn_training_viz.png`
- `results/comparison_viz.png`

## Production Deployment

```bash
# Train before deployment (recommended)
python main.py --mode=train-all --episodes=500 --users=8 --difficulty=medium

# Start real-time session
python main.py --mode=realtime --algorithm=ppo

# Or use GUI
python gui_app.py
```