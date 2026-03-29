# UPRYT v2 - Usage Guide

Complete guide to using all features in UPRYT.

## Quick Reference

| Command | Description |
|---------|-------------|
| `python gui_app.py` | GUI application |
| `python main.py --mode=combined --audio` | Full system |
| `python main.py --mode=train --algorithm=ppo` | Train PPO |
| `python main.py --mode=compare` | Benchmark |

## Modes

### Combined Mode (Recommended)
Full system: posture + attention + hands + dashboard
```bash
python main.py --mode=combined --algorithm=ppo --audio
```

### Real-Time Posture
```bash
python main.py --mode=realtime --algorithm=ppo
python main.py --mode=realtime --algorithm=rule --skip-calibration
```

### Standalone Trackers
```bash
python main.py --mode=hand        # Hand/typing posture
python main.py --mode=attention   # Face/gaze tracking
```

### Training
```bash
python main.py --mode=train --algorithm=ppo --episodes=500
python main.py --mode=train --algorithm=dqn --episodes=500 --enhanced-training
python main.py --mode=train-all --episodes=500
python main.py --mode=compare
```

## All CLI Options

```
--mode={combined,realtime,hand,attention,train,train-all,compare}
--algorithm={ppo,dqn,rule}
--camera=N           Camera index (default: 0)
--episodes=N         Training episodes (default: 500)
--users=N            Simulated users (default: 5)

Feature Flags:
--audio             Audio alerts
--attention         Attention tracking
--hands             Hand tracking
--dashboard         Dashboard export
--online-learning   Online learning
--skip-calibration  Skip calibration
--enhanced-training Curriculum + domain randomization
```

## GUI Application

Run `python gui_app.py` for:

1. **Mode Selection**: Combined, Realtime, Hand, Attention, Train, Compare
2. **Real-Time Options**: Algorithm, camera, audio, calibration
3. **Training Options**: Algorithm, episodes, users, enhanced training
4. **Additional Features**: Attention, hands, dashboard, online learning

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

### Enhanced Training
- Curriculum learning (Easy → Medium → Hard)
- Domain randomization (state noise)
- Simulates fatigue/motivation

## Python API Examples

### Unified Analyzer
```python
from combined_analyzer import create_unified_analyzer

analyzer = create_unified_analyzer(enable_attention=True, enable_hands=True)
metrics = analyzer.analyze(frame, posture_score=0.8, posture_label="good")
print(f"Score: {metrics.combined_score}")
```

### Web Dashboard
```python
from web_dashboard import create_dashboard_exporter

exporter = create_dashboard_exporter()
exporter.start_session("session_001", "user_123")
exporter.record_frame(posture_score=0.85, posture_label="good")
weekly = exporter.get_weekly_summary("user_123")
```

### Enhanced Training
```python
from simulation_enhanced import EnhancedUnifiedTrainer

trainer = EnhancedUnifiedTrainer()
stats = trainer.train(episodes=500, algorithm="ppo")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not working | Try `--camera=1` |
| Low video quality | Use 1080p webcam |
| No models | Run `--mode=train` first |
| Laggy display | Disable unused tracking features |

## Mobile App

See `mobile_app/README.md` for building Android APK.
