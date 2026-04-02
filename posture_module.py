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
    suggestion: str = ""


POSTURE_SUGGESTIONS = {
    PostureLabel.GOOD: [
        "Great posture! Keep it up!",
        "You're sitting beautifully.",
        "Perfect alignment - well done!",
    ],
    PostureLabel.SLOUCHING: [
        "Sit up straight - roll your shoulders back",
        "Engage your core and straighten your spine",
        "Don't slouch - lift your chest and pull shoulders back",
        "Imagine a string pulling you up from the crown of your head",
    ],
    PostureLabel.FORWARD_HEAD: [
        "Move your head back - your ears should align with shoulders",
        "Chin up! Pull your head back slightly",
        "Extend your neck backwards gently",
        "Screen should be at eye level - lower it or raise your chair",
    ],
    PostureLabel.LEANING: [
        "Sit back in your chair evenly",
        "Keep your weight balanced on both hips",
        "Center yourself on your seat",
        "Avoid leaning to one side - distribute weight equally",
    ],
}


class PostureClassifier:
    def __init__(self, baseline: Optional[Dict] = None):
        self.baseline = baseline
        self.history: List[PostureResult] = []
        self.max_history = 50

    def set_baseline(self, baseline: Dict):
        self.baseline = baseline

    def get_suggestion(self, label: PostureLabel, features: Dict) -> str:
        import random
        if label == PostureLabel.UNKNOWN or label == PostureLabel.GOOD:
            return random.choice(POSTURE_SUGGESTIONS[PostureLabel.GOOD]) if label == PostureLabel.GOOD else "No pose detected"
        
        suggestions = POSTURE_SUGGESTIONS.get(label, ["Adjust your posture"])
        
        if label == PostureLabel.FORWARD_HEAD:
            forward_val = features.get("forward_head_y", 0)
            if forward_val > 30:
                suggestions = ["CRITICAL: Move head back NOW - severe forward head posture", "Head is way too forward - pull back immediately!"]
            elif forward_val > 20:
                suggestions = ["Head is drifting forward - bring it back", "Your chin is poking out - tuck it in"]
        
        elif label == PostureLabel.SLOUCHING:
            spine = abs(features.get("spine_inclination", 0))
            if spine > 20:
                suggestions = ["Your spine is curved! Straighten up immediately!", "Major slouch detected - sit up tall!"]
            elif spine > 10:
                suggestions = ["Slight slouch - lift your chest", "Roll your shoulders back"]
        
        elif label == PostureLabel.LEANING:
            shoulder_diff = features.get("shoulder_diff", 0)
            if shoulder_diff > 25:
                suggestions = ["You're leaning heavily to one side!", "Shift your weight to center"]
            elif shoulder_diff > 15:
                suggestions = ["Slight lean detected - balance your weight", "Even out your hips"]
        
        return random.choice(suggestions)

    def classify(self, features: Dict) -> Tuple[PostureLabel, float]:
        if not features or not self.baseline:
            return PostureLabel.UNKNOWN, 0.0
        scores = self._compute_posture_scores(features)
        total_score = self._compute_weighted_score(scores)
        label = self._determine_label(features, scores)
        suggestion = self.get_suggestion(label, features)
        result = PostureResult(label=label, score=total_score, confidence=self._compute_confidence(scores), features=features, details=scores, suggestion=suggestion)
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
        if scores["spine_inclination"] < 0.5 and abs(features.get("spine_inclination", 0)) > config.posture.spine_inclination_threshold * 1.2:
            return PostureLabel.LEANING
        if scores["neck_angle"] < 0.5 and scores["spine_inclination"] < 0.5:
            return PostureLabel.SLOUCHING
        avg_score = np.mean(list(scores.values()))
        if avg_score >= 0.65:
            return PostureLabel.GOOD
        elif avg_score >= 0.45:
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

    def get_suggestion(self, label: PostureLabel, features: Dict) -> str:
        import random
        if label == PostureLabel.UNKNOWN:
            return "No pose detected"
        if label == PostureLabel.GOOD:
            return random.choice(["Great posture! Keep it up!", "Perfect! Stay aligned", "Excellent form!"])

        if label == PostureLabel.FORWARD_HEAD:
            forward_val = features.get("forward_head_y", 0)
            neck_angle = features.get("neck_angle", 90)
            head_tilt = features.get("head_tilt", 0)
            if forward_val > 30:
                return "CRITICAL: Move head back NOW - severe forward head posture"
            elif forward_val > 20:
                if neck_angle > 95:
                    return "Tilt chin DOWN slightly - you're overcorrecting forward"
                elif neck_angle < 85:
                    return "Pull head BACK and UP - extend your neck"
                elif abs(head_tilt) > 5:
                    direction = "right" if head_tilt > 0 else "left"
                    return f"Tilt head to the {direction} - lateral tilt detected"
                else:
                    return "Move head straight BACK toward screen"
            return "Tuck your chin in"

        if label == PostureLabel.SLOUCHING:
            spine = abs(features.get("spine_inclination", 0))
            shoulder_diff = features.get("shoulder_diff", 0)
            if spine > 20:
                return "Your spine is curved! Straighten up immediately!"
            elif shoulder_diff > 15:
                return "One shoulder is higher - lift the lower one"
            return "Lift your chest, roll shoulders back"

        if label == PostureLabel.LEANING:
            shoulder_diff = features.get("shoulder_diff", 0)
            if shoulder_diff > 20:
                return "You're leaning heavily to one side - shift your weight"
            return "Sit upright, don't lean"

        return "Adjust your posture"

    def classify(self, features: Dict) -> Tuple[PostureLabel, float]:
        if not features:
            return PostureLabel.UNKNOWN, 0.0

        # Use baseline-adjusted scoring if available
        baseline = self.baseline
        neck = features.get("neck_angle", 90)
        shoulder_diff = features.get("shoulder_diff", 0)
        shoulder_align = features.get("shoulder_alignment", 0)
        spine = features.get("spine_inclination", 0)
        forward = features.get("forward_head_y", 0)

        if baseline:
            # Adjust thresholds based on user's calibrated good posture
            neck_baseline = baseline.get("neck_angle", 90)
            shoulder_baseline = baseline.get("shoulder_diff", 5)
            spine_baseline = baseline.get("spine_inclination", 0)
            forward_baseline = baseline.get("forward_head_y", 15)

            # Compute deviation from user's baseline (calibrated good posture)
            neck_dev = abs(neck - neck_baseline)
            shoulder_dev = abs(shoulder_diff - shoulder_baseline)
            spine_dev = abs(spine - spine_baseline)
            forward_dev = forward - forward_baseline if forward > forward_baseline else 0
        else:
            neck_dev = abs(neck - 90)
            shoulder_dev = shoulder_diff
            spine_dev = abs(spine)
            forward_dev = forward

        neck_score = max(0, 1 - neck_dev / config.posture.neck_angle_threshold)
        shoulder_score = max(0, 1 - shoulder_dev / config.posture.shoulder_diff_threshold)
        spine_score = max(0, 1 - spine_dev / config.posture.spine_inclination_threshold)
        forward_score = max(0, 1 - forward_dev / config.posture.forward_head_threshold)

        # Include head_tilt in scoring if available
        head_tilt = features.get("head_tilt", 0)
        if head_tilt > 0:
            tilt_score = max(0, 1 - head_tilt / 15.0)
            # Blend tilt into overall score
            neck_score = 0.8 * neck_score + 0.2 * tilt_score

        avg_score = (neck_score + shoulder_score + spine_score + forward_score) / 4

        if avg_score >= 0.65:
            label = PostureLabel.GOOD
        elif forward_dev > config.posture.forward_head_threshold * 0.9:
            label = PostureLabel.FORWARD_HEAD
        elif abs(spine_dev) > config.posture.spine_inclination_threshold * 1.0:
            label = PostureLabel.LEANING
        elif neck_score < 0.6 or shoulder_score < 0.6:
            label = PostureLabel.SLOUCHING
        else:
            label = PostureLabel.GOOD

        return label, avg_score


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
