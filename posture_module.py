import numpy as np
from typing import Dict, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass
from config import config


class PostureLabel(Enum):
    GOOD = "good"
    SLOUCHING = "slouching"
    FORWARD_HEAD = "forward_head"
    LEANING = "leaning"
    UNKNOWN = "unknown"


LABEL_ENCODINGS = {
    PostureLabel.GOOD: 0, PostureLabel.SLOUCHING: 1,
    PostureLabel.FORWARD_HEAD: 2, PostureLabel.LEANING: 3, PostureLabel.UNKNOWN: 4,
}


@dataclass
class PostureResult:
    label: PostureLabel
    score: float
    confidence: float
    features: Dict[str, float]
    details: Dict[str, any]


class PostureClassifier:
    def __init__(self, baseline: Optional[Dict] = None):
        self.baseline = baseline
        self.history: List[PostureResult] = []
        self.max_history = 50

    def set_baseline(self, baseline: Dict):
        self.baseline = baseline

    def classify(self, features: Dict) -> Tuple[PostureLabel, float]:
        if not features or not self.baseline:
            return PostureLabel.UNKNOWN, 0.0
        scores = self._compute_posture_scores(features)
        total_score = self._compute_weighted_score(scores)
        label = self._determine_label(features, scores)
        result = PostureResult(label=label, score=total_score, confidence=self._compute_confidence(scores), features=features, details=scores)
        self._add_to_history(result)
        return label, total_score

    def _compute_posture_scores(self, features: Dict) -> Dict[str, float]:
        return {
            "neck_angle": self._score_neck_angle(features.get("neck_angle", 0)),
            "shoulder_align": self._score_shoulder_alignment(features.get("shoulder_diff", 100)),
            "spine_inclination": self._score_spine_inclination(features.get("spine_inclination", 0)),
            "symmetry": self._score_symmetry(features),
        }

    def _score_neck_angle(self, angle: float) -> float:
        threshold = config.posture.neck_angle_threshold
        deviation = abs(angle - 90)
        score = max(0, 1 - deviation / (threshold * 2))
        return min(1.0, max(0.0, score))

    def _score_shoulder_alignment(self, diff: float) -> float:
        threshold = config.posture.shoulder_diff_threshold
        score = max(0, 1 - diff / (threshold * 2))
        return min(1.0, max(0.0, score))

    def _score_spine_inclination(self, angle: float) -> float:
        threshold = config.posture.spine_inclination_threshold
        deviation = abs(angle)
        score = max(0, 1 - deviation / (threshold * 2))
        return min(1.0, max(0.0, score))

    def _score_symmetry(self, features: Dict) -> float:
        return self._score_shoulder_alignment(features.get("shoulder_diff", 0))

    def _compute_weighted_score(self, scores: Dict) -> float:
        weights = config.posture.score_weights
        total = sum(weights[k] * scores.get(k, 0) for k in weights)
        return min(1.0, max(0.0, total))

    def _compute_confidence(self, scores: Dict) -> float:
        variance = np.var(list(scores.values()))
        return 1.0 - min(1.0, variance * 5)

    def _determine_label(self, features: Dict, scores: Dict) -> PostureLabel:
        forward_threshold = config.posture.forward_head_threshold
        if features.get("forward_head_y", 0) > forward_threshold:
            return PostureLabel.FORWARD_HEAD
        if scores["spine_inclination"] < 0.5 and abs(features.get("spine_inclination", 0)) > config.posture.spine_inclination_threshold:
            return PostureLabel.LEANING
        if scores["neck_angle"] < 0.6 and scores["spine_inclination"] < 0.6:
            return PostureLabel.SLOUCHING
        avg_score = np.mean(list(scores.values()))
        if avg_score >= 0.7:
            return PostureLabel.GOOD
        elif avg_score >= 0.5:
            return PostureLabel.SLOUCHING
        return PostureLabel.LEANING

    def _add_to_history(self, result: PostureResult):
        self.history.append(result)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_recent_trend(self, window: int = 10) -> str:
        if len(self.history) < 2:
            return "stable"
        recent = self.history[-window:]
        scores = [r.score for r in recent]
        if len(scores) < 2:
            return "stable"
        slope = np.polyfit(range(len(scores)), scores, 1)[0]
        if slope > 0.02:
            return "improving"
        elif slope < -0.02:
            return "worsening"
        return "stable"

    def get_average_score(self, window: int = 30) -> float:
        if not self.history:
            return 0.0
        recent = self.history[-window:]
        return np.mean([r.score for r in recent])


class AdaptiveThresholdClassifier(PostureClassifier):
    def __init__(self, baseline: Optional[Dict] = None):
        super().__init__(baseline)
        self.adaptive_weights = config.posture.score_weights.copy()
        self.learning_rate = 0.01

    def update_thresholds(self, user_feedback: float):
        adjustment = user_feedback * self.learning_rate
        self.adaptive_weights["neck_angle"] = min(1.0, self.adaptive_weights["neck_angle"] + adjustment * 0.3)
        self.adaptive_weights["shoulder_align"] = min(1.0, self.adaptive_weights["shoulder_align"] + adjustment * 0.25)
        self.adaptive_weights["spine_inclination"] = min(1.0, self.adaptive_weights["spine_inclination"] + adjustment * 0.25)
        self.adaptive_weights["symmetry"] = min(1.0, self.adaptive_weights["symmetry"] + adjustment * 0.2)

    def get_personalized_weights(self) -> Dict:
        return self.adaptive_weights.copy()


class RuleBasedClassifier:
    def __init__(self):
        self.baseline = None

    def set_baseline(self, baseline: Dict):
        self.baseline = baseline

    @staticmethod
    def classify(features: Dict) -> Tuple[PostureLabel, float]:
        if not features:
            return PostureLabel.UNKNOWN, 0.0
        score = 0
        count = 0
        neck = features.get("neck_angle", 90)
        if 85 <= neck <= 95:
            score += 1
        count += 1
        shoulder_diff = features.get("shoulder_diff", 0)
        if shoulder_diff < config.posture.shoulder_diff_threshold:
            score += 1
        count += 1
        spine = features.get("spine_inclination", 0)
        if abs(spine) < config.posture.spine_inclination_threshold:
            score += 1
        count += 1
        forward = features.get("forward_head_y", 0)
        if forward < config.posture.forward_head_threshold:
            score += 1
        count += 1
        normalized_score = score / count if count > 0 else 0.0
        if score >= 3:
            label = PostureLabel.GOOD
        elif score == 2:
            label = PostureLabel.SLOUCHING
        elif forward >= config.posture.forward_head_threshold:
            label = PostureLabel.FORWARD_HEAD
        else:
            label = PostureLabel.LEANING
        return label, normalized_score


def encode_label(label: PostureLabel) -> int:
    return LABEL_ENCODINGS.get(label, 4)


def decode_label(encoding: int) -> PostureLabel:
    for label, code in LABEL_ENCODINGS.items():
        if code == encoding:
            return label
    return PostureLabel.UNKNOWN


if __name__ == "__main__":
    classifier = RuleBasedClassifier()
    test_features = {"neck_angle": 92, "shoulder_diff": 8, "spine_inclination": 5, "forward_head_y": 10}
    label, score = classifier.classify(test_features)
    print(f"Label: {label.value}, Score: {score:.2f}")
