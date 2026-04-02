# UPRYT v2 - Usage Guide

Complete guide to using all features in UPRYT v2.

## Quick Reference

| Command | Description |
|---------|-------------|
| `python gui_app.py` | GUI application |
| `python main.py --mode combined --algorithm ppo --audio` | Full system |
| `python main.py --mode realtime --algorithm auto` | Posture only with auto-switching |
| `python main.py --mode train --algorithm ppo --episodes 2000 --enhanced-training` | Train PPO |
| `python main.py --mode train --algorithm dqn --episodes 2000 --enhanced-training` | Train DQN |
| `python main.py --mode train-all --episodes 2000 --enhanced-training` | Train both |
| `python main.py --mode compare` | Benchmark algorithms |
| `python visualize_results.py --benchmark` | Visualize training results |

## Modes

### Combined Mode (Recommended)
Full system: posture + attention + hands with automatic algorithm switching
```bash
python main.py --mode combined --algorithm auto --audio
# Or with specific algorithm
python main.py --mode combined --algorithm ppo --audio
```

### Real-Time Posture
```bash
# With auto-switching (default)
python main.py --mode realtime --algorithm auto

# With specific algorithm
python main.py --mode realtime --algorithm ppo
python main.py --mode realtime --algorithm dqn
python main.py --mode realtime --algorithm rule --skip-calibration
```

### Standalone Trackers
```bash
python main.py --mode hand        # Hand/typing posture
python main.py --mode attention   # Face/gaze tracking
```

### Training
```bash
# Train PPO (recommended)
python main.py --mode train --algorithm ppo --episodes 2000 --enhanced-training --difficulty medium

# Train DQN
python main.py --mode train --algorithm dqn --episodes 2000 --enhanced-training --difficulty medium

# Train both PPO and DQN
python main.py --mode train-all --episodes 2000 --enhanced-training --difficulty medium

# Benchmark comparison
python main.py --mode compare
```

### Visualization
```bash
# Auto-detect latest training runs
python visualize_results.py

# With benchmark metrics (requires trained models)
python visualize_results.py --benchmark

# Custom files
python visualize_results.py --ppo-file results/ppo_20260402_123456.json --dqn-file results/dqn_20260402_123456.json

# Adjust moving average window
python visualize_results.py --window 20 --benchmark
```

## All CLI Options

```
--mode={combined,realtime,hand,attention,train,train-all,compare}
--algorithm={ppo,dqn,rule,auto}    # auto = automatic algorithm selection

--camera=N           Camera index (default: 0)
--episodes=N        Training episodes (default: 500)
--users=N           Simulated users (default: 5)
--difficulty={easy,medium,hard}  Training difficulty (default: medium)

Feature Flags:
--audio             Audio alerts
--attention         Attention tracking
--hands             Hand tracking
--skip-calibration  Skip calibration
--enhanced-training Use curriculum + domain randomization (recommended for training)
--auto-switch       Enable automatic algorithm switching (default: on)
--no-auto-switch    Disable automatic algorithm switching
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
- Good posture maintained: +0.05 to +0.2
- Bad posture ignored: -0.1

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

## Training Visualization

The visualization tool generates:

| File | Description |
|------|-------------|
| `ppo_training_viz.png` | Episode rewards, eval rewards, distribution, stats |
| `dqn_training_viz.png` | Same as PPO for DQN |
| `comparison_viz.png` | PPO vs DQN comparison (final, best, avg, MA) |
| `benchmark_viz.png` | Correction rate, avg reward, total alerts |

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

# Train PPO
stats = train_ppo(num_episodes=2000, num_users=8, verbose=True, difficulty='medium')

# Train DQN
stats = train_dqn(num_episodes=2000, num_users=8, verbose=True, difficulty='medium')
```

### Benchmark
```python
from simulation_enhanced import compare_algorithms
from rl_ppo_agent import PPOPPOAgent
from rl_agent import DQNAgent

results = compare_algorithms(
    ppo_path='models/ppo_best.pth',
    dqn_path='models/dqn_best.pth',
    num_episodes=50
)
print(results)
```

### Visualization
```python
# Run after training to generate plots
import subprocess
subprocess.run(['python', 'visualize_results.py', '--benchmark'])
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not working | Try `--camera=1` |
| Low video quality | Use 1080p webcam |
| No trained models | Run `--mode train --algorithm ppo --episodes 500 --enhanced-training` |
| Laggy display | Disable unused tracking features (--attention, --hands) |
| Training unstable | Try `--difficulty easy` first |
| Poor correction rate | Train for more episodes (2000+), try hard difficulty |

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
config.system.decision_interval = 2.5
config.system.calibration_frames = 20

# Posture Thresholds
config.posture.neck_angle_threshold = 20.0
config.posture.shoulder_diff_threshold = 20.0
```

## Files Generated

### Training
- `results/ppo_{timestamp}.json` - PPO training data
- `results/dqn_{timestamp}.json` - DQN training data
- `models/ppo_*.pth` - PPO model checkpoints
- `models/dqn_*.pth` - DQN model checkpoints

### Visualization
- `results/ppo_training_viz.png`
- `results/dqn_training_viz.png`
- `results/comparison_viz.png`
- `results/benchmark_viz.png`