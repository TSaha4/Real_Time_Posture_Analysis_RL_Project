import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import threading


@dataclass
class TrainingMetrics:
    episode: int
    reward: float
    loss: float
    epsilon: float
    avg_score: float
    correction_rate: float
    timestamp: float


class MetricsLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_file = os.path.join(log_dir, "training_metrics.jsonl")
        self.session_start = time.time()
        self.metrics_buffer: List[TrainingMetrics] = []
        self.lock = threading.Lock()

    def log_episode(self, episode: int, reward: float, loss: float, epsilon: float,
                   avg_score: float = 0.0, correction_rate: float = 0.0):
        metric = TrainingMetrics(
            episode=episode, reward=reward, loss=loss, epsilon=epsilon,
            avg_score=avg_score, correction_rate=correction_rate,
            timestamp=time.time() - self.session_start)
        with self.lock:
            self.metrics_buffer.append(metric)
        if len(self.metrics_buffer) >= 10:
            self._flush_buffer()

    def _flush_buffer(self):
        with self.lock:
            if not self.metrics_buffer:
                return
            with open(self.metrics_file, "a") as f:
                for metric in self.metrics_buffer:
                    f.write(json.dumps(asdict(metric)) + "\n")
            self.metrics_buffer = []

    def get_recent_metrics(self, n: int = 100) -> List[TrainingMetrics]:
        if not os.path.exists(self.metrics_file):
            return []
        metrics = []
        with open(self.metrics_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    metrics.append(TrainingMetrics(**data))
                except:
                    continue
        return metrics[-n:]

    def compute_statistics(self, window: int = 100) -> Dict[str, float]:
        recent = self.get_recent_metrics(window)
        if not recent:
            return {"avg_reward": 0.0, "avg_loss": 0.0, "avg_epsilon": 0.0, "avg_score": 0.0, "avg_correction_rate": 0.0}
        return {
            "avg_reward": np.mean([m.reward for m in recent]),
            "avg_loss": np.mean([m.loss for m in recent if m.loss > 0]),
            "avg_epsilon": np.mean([m.epsilon for m in recent]),
            "avg_score": np.mean([m.avg_score for m in recent]),
            "avg_correction_rate": np.mean([m.correction_rate for m in recent]),
        }

    def close(self):
        self._flush_buffer()


class SessionLogger:
    def __init__(self, session_name: str = None, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name
        self.session_file = os.path.join(log_dir, f"session_{session_name}.json")
        self.data: Dict[str, Any] = {
            "session_name": session_name, "start_time": datetime.now().isoformat(),
            "end_time": None, "total_runtime": 0, "alerts_sent": 0,
            "corrections_made": 0, "posture_scores": [], "actions_taken": [], "rewards": [],
        }

    def log_frame(self, posture_score: float, action: int, reward: float):
        self.data["posture_scores"].append(posture_score)
        self.data["actions_taken"].append(action)
        self.data["rewards"].append(reward)
        if action != 0:
            self.data["alerts_sent"] += 1

    def log_correction(self):
        self.data["corrections_made"] += 1

    def finalize(self):
        self.data["end_time"] = datetime.now().isoformat()
        if self.data["start_time"]:
            start = datetime.fromisoformat(self.data["start_time"])
            end = datetime.now()
            self.data["total_runtime"] = (end - start).total_seconds()
        with open(self.session_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "session_name": self.data["session_name"], "total_runtime": self.data["total_runtime"],
            "total_frames": len(self.data["posture_scores"]),
            "alerts_sent": self.data["alerts_sent"], "corrections_made": self.data["corrections_made"],
        }
        if self.data["posture_scores"]:
            scores = self.data["posture_scores"]
            summary["avg_posture_score"] = np.mean(scores)
            summary["max_posture_score"] = np.max(scores)
            summary["min_posture_score"] = np.min(scores)
        if self.data["rewards"]:
            summary["total_reward"] = np.sum(self.data["rewards"])
            summary["avg_reward"] = np.mean(self.data["rewards"])
        return summary


def download_mediapipe_model(model_name: str = "pose_landmarker_lite.task"):
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    if os.path.exists(model_name):
        print(f"Model {model_name} already exists")
        return True
    print(f"Downloading {model_name}...")
    try:
        urllib.request.urlretrieve(url, model_name)
        print(f"Downloaded {model_name} successfully")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False


def ensure_mediapipe_models():
    if not os.path.exists("pose_landmarker_lite.task"):
        print("Model not found, downloading...")
        download_mediapipe_model()


def setup_directories():
    for d in ["logs", "models", "data"]:
        os.makedirs(d, exist_ok=True)


def validate_config():
    from config import config
    errors = []
    if config.rl.state_size <= 0:
        errors.append("RL state_size must be positive")
    if config.rl.action_size <= 0:
        errors.append("RL action_size must be positive")
    if not 0 < config.rl.gamma <= 1:
        errors.append("RL gamma must be between 0 and 1")
    if not 0 < config.rl.epsilon_start <= 1:
        errors.append("RL epsilon_start must be between 0 and 1")
    if config.system.decision_interval <= 0:
        errors.append("System decision_interval must be positive")
    if errors:
        for error in errors:
            print(f"Config error: {error}")
        return False
    return True


def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.degrees(np.arccos(cos_angle))


def normalize_angle(angle: float) -> float:
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def moving_average(data: List[float], window: int = 10) -> List[float]:
    if len(data) < window:
        return data
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(np.mean(data[start:i+1]))
    return result


def exponential_moving_average(data: List[float], alpha: float = 0.1) -> List[float]:
    if not data:
        return []
    result = [data[0]]
    for value in data[1:]:
        ema = alpha * value + (1 - alpha) * result[-1]
        result.append(ema)
    return result


class RateCalculator:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.timestamps: List[float] = []
        self.start_time = time.time()

    def tick(self):
        self.timestamps.append(time.time() - self.start_time)
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

    def get_fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed == 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


if __name__ == "__main__":
    setup_directories()
    logger = MetricsLogger()
    for i in range(10):
        logger.log_episode(i, reward=i*10, loss=1/i if i > 0 else 0, epsilon=1/(i+1), avg_score=0.8, correction_rate=0.7)
    stats = logger.compute_statistics()
    print(f"Statistics: {stats}")
    logger.close()
    print("Utils test complete")
