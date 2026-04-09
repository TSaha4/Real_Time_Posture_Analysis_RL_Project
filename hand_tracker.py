import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import time
import mediapipe as mp


class TypingPosture(Enum):
    GOOD = "good"
    TENSE = "tense"
    RAISED_ARMS = "raised_arms"
    ASYMMETRIC = "asymmetric"
    UNKNOWN = "unknown"


@dataclass
class HandMetrics:
    left_hand_detected: bool = False
    right_hand_detected: bool = False
    left_hand_height: float = 0.0
    right_hand_height: float = 0.0
    hand_spread: float = 0.0
    typing_intensity: float = 0.0
    symmetry_score: float = 1.0
    posture_state: TypingPosture = TypingPosture.UNKNOWN
    tension_score: float = 0.0


class HandTracker:
    def __init__(self, use_holistic: bool = True):
        self.use_holistic = use_holistic
        
        if use_holistic:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
        
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        
        self._hand_history: List[Dict] = []
        self._typing_motion: List[float] = []
        self._last_hand_positions: Dict[str, Tuple] = {}
        self._session_metrics = {
            "total_time": 0.0,
            "good_typing_time": 0.0,
            "tense_time": 0.0,
            "raised_arms_time": 0.0,
            "asymmetric_time": 0.0,
            "typing_episodes": 0,
        }
        self._is_typing = False
        self._typing_start_time = None
        self._current_state = TypingPosture.UNKNOWN
        self._state_start_time = time.time()

    def detect_hands(self, frame: np.ndarray) -> Optional[Dict]:
        if not self.use_holistic:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
        
        hands_data = {
            "hands": [],
            "left_hand": None,
            "right_hand": None,
        }
        
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, 
            results.multi_handedness
        ):
            hand_type = handedness.classification[0].label.lower()
            
            hand_data = self._extract_hand_features(hand_landmarks, frame.shape)
            hand_data["type"] = hand_type
            hand_data["landmarks"] = hand_landmarks
            
            hands_data["hands"].append(hand_data)
            
            if hand_type == "left":
                hands_data["left_hand"] = hand_data
            else:
                hands_data["right_hand"] = hand_data
        
        return hands_data

    def _extract_hand_features(self, landmarks, frame_shape: Tuple) -> Dict:
        h, w = frame_shape[:2]
        
        wrist = landmarks.landmark[0]
        wrist_pos = (wrist.x * w, wrist.y * h)
        
        finger_tips = [
            landmarks.landmark[4],   # thumb
            landmarks.landmark[8],  # index
            landmarks.landmark[12], # middle
            landmarks.landmark[16], # ring
            landmarks.landmark[20], # pinky
        ]
        
        finger_bases = [
            landmarks.landmark[2],  # thumb base
            landmarks.landmark[5],  # index base
            landmarks.landmark[9], # middle base
            landmarks.landmark[13], # ring base
            landmarks.landmark[17], # pinky base
        ]
        
        avg_finger_height = np.mean([f.y * h for f in finger_tips])
        hand_height = wrist.y * h
        
        spread = 0
        for tip, base in zip(finger_tips, finger_bases):
            spread += abs(tip.x - base.x) * w
        avg_spread = spread / len(finger_tips)
        
        tension = 0
        for tip, base in zip(finger_tips, finger_bases):
            tip_dist = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
            base_dist = np.sqrt((base.x - wrist.x)**2 + (base.y - wrist.y)**2)
            if base_dist > 0:
                tension += abs(tip_dist - base_dist) / base_dist
        avg_tension = tension / len(finger_tips)
        
        motion_score = 0
        hand_id = f"{wrist.x:.2f}_{wrist.y:.2f}"
        if hand_id in self._last_hand_positions:
            last_pos = self._last_hand_positions[hand_id]
            motion = np.sqrt((wrist.x - last_pos[0])**2 + (wrist.y - last_pos[1])**2)
            motion_score = min(1.0, motion * 100)
        
        self._last_hand_positions[hand_id] = (wrist.x, wrist.y)
        
        return {
            "wrist_position": wrist_pos,
            "wrist_height": hand_height,
            "avg_finger_height": avg_finger_height,
            "hand_spread": avg_spread,
            "tension_score": avg_tension,
            "motion_score": motion_score,
        }

    def draw_hands(self, frame: np.ndarray, hands_data: Dict) -> np.ndarray:
        if not hands_data or not hands_data.get("hands"):
            return frame
        
        for hand_landmarks in hands_data["hands"]:
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks["landmarks"],
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw_styles.get_default_hand_landmarks_style(),
                self.mp_draw_styles.get_default_hand_connections_style()
            )
        
        return frame

    def update_metrics(self, hands_data: Optional[Dict]) -> HandMetrics:
        metrics = HandMetrics()
        current_time = time.time()
        frame_duration = current_time - self._state_start_time
        
        self._session_metrics["total_time"] += frame_duration
        self._state_start_time = current_time
        
        if hands_data is None:
            metrics.posture_state = TypingPosture.UNKNOWN
            return metrics
        
        left = hands_data.get("left_hand")
        right = hands_data.get("right_hand")
        
        metrics.left_hand_detected = left is not None
        metrics.right_hand_detected = right is not None
        
        if left:
            metrics.left_hand_height = left["wrist_height"]
        if right:
            metrics.right_hand_height = right["wrist_height"]
        
        if left and right:
            metrics.hand_spread = np.sqrt(
                (left["wrist_position"][0] - right["wrist_position"][0])**2 +
                (left["wrist_position"][1] - right["wrist_position"][1])**2
            )
            
            height_diff = abs(left["wrist_height"] - right["wrist_height"])
            symmetry = max(0, 1.0 - height_diff / 100)
            metrics.symmetry_score = symmetry
            
            avg_tension = (left["tension_score"] + right["tension_score"]) / 2
            metrics.tension_score = avg_tension
            
            avg_motion = (left["motion_score"] + right["motion_score"]) / 2
            metrics.typing_intensity = avg_motion
            
            self._typing_motion.append(avg_motion)
            if len(self._typing_motion) > 50:
                self._typing_motion.pop(0)
            
            current_typing = avg_motion > 0.3
            if current_typing and not self._is_typing:
                self._is_typing = True
                self._typing_start_time = current_time
                self._session_metrics["typing_episodes"] += 1
            elif not current_typing and self._is_typing:
                self._is_typing = False
            
            posture = self._determine_posture(
                left, right, symmetry, avg_tension, avg_motion
            )
            self._update_state(posture, frame_duration, metrics)
            metrics.posture_state = posture
        elif left or right:
            hand = left or right
            metrics.tension_score = hand["tension_score"]
            metrics.typing_intensity = hand["motion_score"]
            self._update_state(TypingPosture.ASYMMETRIC, frame_duration, metrics)
            metrics.posture_state = TypingPosture.ASYMMETRIC
        
        self._hand_history.append({
            "timestamp": current_time,
            "metrics": metrics,
        })
        if len(self._hand_history) > 1000:
            self._hand_history.pop(0)
        
        return metrics

    def _determine_posture(self, left: Dict, right: Dict, symmetry: float,
                          tension: float, motion: float) -> TypingPosture:
        if symmetry < 0.7:
            return TypingPosture.ASYMMETRIC
        
        height_diff = abs(left["wrist_height"] - right["wrist_height"])
        if height_diff > 80:
            return TypingPosture.RAISED_ARMS
        
        if tension > 0.3:
            return TypingPosture.TENSE
        
        if motion > 0.2:
            return TypingPosture.GOOD
        
        return TypingPosture.GOOD

    def _update_state(self, new_state: TypingPosture, frame_duration: float, metrics: HandMetrics):
        if new_state != self._current_state:
            self._current_state = new_state
        
        if new_state == TypingPosture.GOOD:
            self._session_metrics["good_typing_time"] += frame_duration
        elif new_state == TypingPosture.TENSE:
            self._session_metrics["tense_time"] += frame_duration
        elif new_state == TypingPosture.RAISED_ARMS:
            self._session_metrics["raised_arms_time"] += frame_duration
        elif new_state == TypingPosture.ASYMMETRIC:
            self._session_metrics["asymmetric_time"] += frame_duration

    def get_current_state(self) -> TypingPosture:
        return self._current_state

    def is_typing(self) -> bool:
        return self._is_typing

    def get_typing_intensity(self) -> float:
        if not self._typing_motion:
            return 0.0
        return np.mean(self._typing_motion[-10:])

    def get_session_metrics(self) -> Dict:
        total = max(0.1, self._session_metrics["total_time"])
        return {
            "total_time_seconds": self._session_metrics["total_time"],
            "good_typing_time": self._session_metrics["good_typing_time"],
            "tense_time": self._session_metrics["tense_time"],
            "raised_arms_time": self._session_metrics["raised_arms_time"],
            "asymmetric_time": self._session_metrics["asymmetric_time"],
            "typing_episodes": self._session_metrics["typing_episodes"],
            "good_typing_percentage": (self._session_metrics["good_typing_time"] / total) * 100,
            "current_state": self._current_state.value,
        }

    def get_typing_posture_score(self) -> float:
        if not self._hand_history:
            return 1.0
        
        recent = self._hand_history[-50:]
        if not recent:
            return 1.0
        
        good_count = sum(1 for h in recent if h["metrics"].posture_state == TypingPosture.GOOD)
        return good_count / len(recent)

    def reset(self):
        self._hand_history = []
        self._typing_motion = []
        self._last_hand_positions = {}
        self._session_metrics = {
            "total_time": 0.0,
            "good_typing_time": 0.0,
            "tense_time": 0.0,
            "raised_arms_time": 0.0,
            "asymmetric_time": 0.0,
            "typing_episodes": 0,
        }
        self._is_typing = False
        self._typing_start_time = None
        self._current_state = TypingPosture.UNKNOWN
        self._state_start_time = time.time()


class CombinedPostureAnalyzer:
    def __init__(self):
        self.hand_tracker = HandTracker(use_holistic=True)
        self._combined_history: List[Dict] = []

    def analyze(self, frame: np.ndarray, posture_score: float, 
                posture_label: str) -> Dict:
        hands_data = self.hand_tracker.detect_hands(frame)
        hand_metrics = self.hand_tracker.update_metrics(hands_data)
        
        if hands_data:
            frame = self.hand_tracker.draw_hands(frame, hands_data)
        
        posture_posture_score = hand_metrics.symmetry_score * hand_metrics.typing_intensity
        combined_score = posture_score * (0.7 + posture_posture_score * 0.3)
        
        analysis = {
            "posture_score": posture_score,
            "typing_posture_score": combined_score,
            "hand_metrics": {
                "left_detected": hand_metrics.left_hand_detected,
                "right_detected": hand_metrics.right_hand_detected,
                "symmetry_score": hand_metrics.symmetry_score,
                "tension_score": hand_metrics.tension_score,
                "typing_intensity": hand_metrics.typing_intensity,
                "posture_state": hand_metrics.posture_state.value,
            },
            "combined_score": combined_score,
            "is_typing": self.hand_tracker.is_typing(),
            "typing_intensity": self.hand_tracker.get_typing_intensity(),
        }
        
        self._combined_history.append(analysis)
        if len(self._combined_history) > 1000:
            self._combined_history.pop(0)
        
        return analysis

    def get_session_summary(self) -> Dict:
        hand_metrics = self.hand_tracker.get_session_metrics()
        
        if not self._combined_history:
            return {"error": "No data available"}
        
        recent_scores = [c["combined_score"] for c in self._combined_history[-100:]]
        recent_typing = [c["typing_intensity"] for c in self._combined_history[-100:] if c["is_typing"]]
        
        return {
            "hand_metrics": hand_metrics,
            "avg_combined_score": np.mean(recent_scores),
            "typing_episodes": hand_metrics.get("typing_episodes", 0),
            "good_typing_percentage": hand_metrics.get("good_typing_percentage", 0),
            "avg_typing_intensity": np.mean(recent_typing) if recent_typing else 0,
        }


def create_hand_tracker() -> HandTracker:
    return HandTracker()


def create_combined_analyzer() -> CombinedPostureAnalyzer:
    return CombinedPostureAnalyzer()


if __name__ == "__main__":
    print("Testing Hand Tracking for Typing Posture...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
    else:
        tracker = HandTracker(use_holistic=True)
        
        print("Starting hand tracking... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            hands_data = tracker.detect_hands(frame)
            metrics = tracker.update_metrics(hands_data)
            
            if hands_data:
                frame = tracker.draw_hands(frame, hands_data)
            
            cv2.putText(frame, f"State: {metrics.posture_state.value}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Symmetry: {metrics.symmetry_score:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Typing: {tracker.get_typing_intensity():.2f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Hand Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\nSession Summary:")
        print(tracker.get_session_metrics())
