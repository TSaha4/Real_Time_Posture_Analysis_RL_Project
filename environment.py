import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from posture_module import PostureLabel, encode_label
from config import config


class Action(Enum):
    NO_FEEDBACK = 0
    SUBTLE_ALERT = 1
    STRONG_ALERT = 2

    @property
    def name(self) -> str:
        return ["no_feedback", "subtle_alert", "strong_alert"][self.value]


@dataclass
class PostureState:
    posture_label: int
    posture_score: float
    duration_bad_posture: float
    time_since_alert: float
    recent_corrections: List[bool] = field(default_factory=list)
    consecutive_alerts: int = 0

    # NEW: Raw geometric features (for better RL state representation)
    neck_angle: float = 90.0
    shoulder_diff: float = 0.0
    spine_inclination: float = 0.0
    forward_head_y: float = 0.0
    head_tilt: float = 0.0

    # NEW: Trend features (for temporal awareness)
    score_velocity: float = 0.0  # Rate of score change
    posture_trend: float = 0.0   # Improving/worsening
    alert_frequency: float = 0.0  # Alerts per minute
    time_in_bad_posture: float = 0.0

    # NEW: User context (for personalization)
    user_compliance: float = 0.7
    user_fatigue: float = 0.0
    correction_effectiveness: float = 0.5

    def to_array(self) -> np.ndarray:
        corrections_mean = np.mean(self.recent_corrections) if self.recent_corrections else 0.0
        corrections_recent = np.mean(self.recent_corrections[-5:]) if len(self.recent_corrections) >= 5 else corrections_mean
        return np.array([
            float(self.posture_label),
            self.posture_score,
            min(self.duration_bad_posture / 30.0, 1.0),
            min(self.time_since_alert / 30.0, 1.0),
            corrections_recent,
            float(self.consecutive_alerts) / 5.0,
            # Raw features (5)
            (self.neck_angle - 90) / 30.0,  # normalized deviation from ideal
            self.shoulder_diff / 30.0,
            self.spine_inclination / 20.0,
            self.forward_head_y / 30.0,
            self.head_tilt / 15.0,
            # Trend features (4)
            max(-1, min(1, self.score_velocity)),
            max(-1, min(1, self.posture_trend)),
            self.alert_frequency / 5.0,
            min(self.time_in_bad_posture / 60.0, 1.0),
            # User context (3)
            self.user_compliance,
            self.user_fatigue,
            self.correction_effectiveness,
        ], dtype=np.float32)

    def update_trend(self, prev_score: float, prev_time: float):
        """Update trend features based on previous state"""
        if prev_time > 0 and self.time_since_alert > 0:
            score_delta = self.posture_score - prev_score
            time_delta = self.time_since_alert - prev_time
            if time_delta > 0:
                self.score_velocity = score_delta / time_delta
            else:
                self.score_velocity = 0.0

        # Update posture trend (smoothed)
        if len(self.recent_corrections) >= 3:
            recent = self.recent_corrections[-3:]
            self.posture_trend = np.mean(recent) * 2 - 1  # -1 to 1 scale

    @classmethod
    def from_posture(cls, label: PostureLabel, score: float) -> "PostureState":
        return cls(
            posture_label=encode_label(label),
            posture_score=score,
            duration_bad_posture=0.0,
            time_since_alert=0.0,
            recent_corrections=[],
            consecutive_alerts=0,
            neck_angle=90.0,
            shoulder_diff=0.0,
            spine_inclination=0.0,
            forward_head_y=15.0,
            head_tilt=0.0,
        )


@dataclass
class EnvironmentMetrics:
    total_episodes: int = 0
    total_alerts: int = 0
    successful_corrections: int = 0
    cumulative_reward: float = 0.0
    avg_posture_score: float = 0.0
    alert_fatigue_count: int = 0
    episode_rewards: List[float] = field(default_factory=list)
    episode_durations: List[float] = field(default_factory=list)


class PostureEnvironment:
    def __init__(self):
        self.current_state: Optional[PostureState] = None
        self.last_action: Optional[Action] = None
        self.episode_reward: float = 0.0
        self.episode_step: int = 0
        self.metrics = EnvironmentMetrics()
        self.alert_history: List[float] = []
        self.posture_history: List[float] = []

    def reset(self) -> PostureState:
        self.current_state = PostureState(
            posture_label=encode_label(PostureLabel.GOOD),
            posture_score=0.7,
            duration_bad_posture=0.0,
            time_since_alert=0.0,
            recent_corrections=[],
            consecutive_alerts=0,
        )
        self.last_action = None
        self.episode_reward = 0.0
        self.episode_step = 0
        return self.current_state

    def step(self, action: int, new_label: PostureLabel = None, new_score: float = None) -> Tuple[PostureState, float, bool]:
        action_enum = Action(action)
        if self.current_state is None:
            self.reset()
        self.current_state.time_since_alert += config.system.decision_interval
        if action_enum == Action.NO_FEEDBACK:
            self.current_state.consecutive_alerts = 0
            reward = self._compute_reward_no_action(new_score)
        elif action_enum == Action.SUBTLE_ALERT:
            reward = self._compute_reward_subtle_alert(new_label, new_score)
        else:
            reward = self._compute_reward_strong_alert(new_label, new_score)
        self._update_posture_state(new_label, new_score)
        self._update_time_since_alert(action_enum)
        self._update_posture_duration(new_label)
        self.episode_reward += reward
        self.episode_step += 1
        self.metrics.cumulative_reward += reward
        done = self.episode_step >= 500
        if done:
            self._finish_episode()
        return self.current_state, reward, done

    def _compute_reward_no_action(self, new_score: float) -> float:
        if new_score is None:
            return config.reward.no_face_detected

        prev_score = self.current_state.posture_score if self.current_state else 0.7

        score_delta = new_score - prev_score
        if score_delta > 0.01:
            reward = 2.0 * score_delta
        elif score_delta < -0.01:
            reward = 3.0 * score_delta
        else:
            # Penalty for sustained bad posture (missed opportunity to correct)
            if new_score < 0.5:
                reward = -0.3
            else:
                reward = config.reward.sustained_good

        return reward

    def _compute_reward_subtle_alert(self, new_label: PostureLabel, new_score: float) -> float:
        if new_label is None:
            return config.reward.no_face_detected
        self.current_state.consecutive_alerts += 1
        self.current_state.time_since_alert = 0.0
        self.metrics.total_alerts += 1

        if new_label == PostureLabel.GOOD or (new_score and new_score >= 0.7):
            self.current_state.recent_corrections.append(True)
            self.metrics.successful_corrections += 1
            reward = config.reward.posture_improve * 0.8  # Reduced from 10.0 to 8.0
            if self.current_state.consecutive_alerts == 1:
                reward += 2.0
        else:
            self.current_state.recent_corrections.append(False)
            reward = config.reward.alert_ignored * 0.8
            if self.current_state.duration_bad_posture < 5.0:
                reward += 0.5

        return reward

    def _compute_reward_strong_alert(self, new_label: PostureLabel, new_score: float) -> float:
        if new_label is None:
            return config.reward.no_face_detected
        self.current_state.consecutive_alerts += 1
        self.current_state.time_since_alert = 0.0
        self.metrics.total_alerts += 1

        if self.current_state.consecutive_alerts > 3:
            self.metrics.alert_fatigue_count += 1

        if new_label == PostureLabel.GOOD or (new_score and new_score >= 0.7):
            self.current_state.recent_corrections.append(True)
            self.metrics.successful_corrections += 1
            reward = config.reward.posture_improve
            if self.current_state.consecutive_alerts == 1:
                reward += 1.5
        else:
            self.current_state.recent_corrections.append(False)
            reward = config.reward.alert_ignored
            if self.current_state.duration_bad_posture < 5.0:
                reward += 0.5

        return reward

    def _update_posture_state(self, label: PostureLabel, score: float):
        if label is not None:
            self.current_state.posture_label = encode_label(label)
        if score is not None:
            self.current_state.posture_score = score
        if len(self.current_state.recent_corrections) > 20:
            self.current_state.recent_corrections.pop(0)

    def _update_time_since_alert(self, action: Action):
        if action != Action.NO_FEEDBACK:
            self.current_state.time_since_alert = 0.0

    def _update_posture_duration(self, label: PostureLabel):
        if label is not None and label != PostureLabel.GOOD:
            self.current_state.duration_bad_posture += config.system.decision_interval
        else:
            self.current_state.duration_bad_posture = max(0, self.current_state.duration_bad_posture - 1)

    def _finish_episode(self):
        self.metrics.total_episodes += 1
        self.metrics.episode_rewards.append(self.episode_reward)
        self.metrics.episode_durations.append(self.episode_step)
        if len(self.posture_history) > 0:
            self.metrics.avg_posture_score = np.mean(self.posture_history)

    def get_state(self) -> Optional[PostureState]:
        return self.current_state

    def get_state_array(self) -> np.ndarray:
        if self.current_state is None:
            return np.zeros(config.rl.state_size, dtype=np.float32)
        return self.current_state.to_array()

    def get_metrics(self) -> Dict:
        return {
            "total_episodes": self.metrics.total_episodes,
            "total_alerts": self.metrics.total_alerts,
            "successful_corrections": self.metrics.successful_corrections,
            "cumulative_reward": self.metrics.cumulative_reward,
            "avg_posture_score": self.metrics.avg_posture_score,
            "alert_fatigue_count": self.metrics.alert_fatigue_count,
            "correction_rate": self._get_correction_rate(),
            "avg_episode_reward": np.mean(self.metrics.episode_rewards[-100:]) if self.metrics.episode_rewards else 0,
        }

    def _get_correction_rate(self) -> float:
        if self.metrics.total_alerts == 0:
            return 0.0
        return self.metrics.successful_corrections / self.metrics.total_alerts

    def add_posture_observation(self, label: PostureLabel, score: float):
        self.posture_history.append(score)
        if len(self.posture_history) > 1000:
            self.posture_history.pop(0)


class RuleBasedEnvironment:
    def __init__(self):
        self.alert_cooldown = config.system.cooldown_period
        self.last_alert_time = 0.0

    def should_alert(self, label: PostureLabel, duration: float, current_time: float) -> Tuple[bool, int]:
        is_bad = label != PostureLabel.GOOD and label != PostureLabel.UNKNOWN
        if not is_bad:
            return False, Action.NO_FEEDBACK.value
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False, Action.NO_FEEDBACK.value
        if duration > 10:
            self.last_alert_time = current_time
            return True, Action.STRONG_ALERT.value
        elif duration > 3:
            self.last_alert_time = current_time
            return True, Action.SUBTLE_ALERT.value
        return False, Action.NO_FEEDBACK.value


if __name__ == "__main__":
    env = PostureEnvironment()
    state = env.reset()
    print(f"Initial state shape: {state.to_array().shape}")
    for i in range(10):
        action = np.random.randint(0, 3)
        next_state, reward, done = env.step(action, PostureLabel.GOOD, 0.8)
        print(f"Step {i}: Action={action}, Reward={reward:.2f}, Done={done}")
