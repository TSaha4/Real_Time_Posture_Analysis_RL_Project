import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import time


class AttentionState(Enum):
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    AWAY = "away"
    UNKNOWN = "unknown"


@dataclass
class GazeMetrics:
    eye_aspect_ratio: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    mouth_open_ratio: float = 0.0
    blink_rate: float = 0.0
    attention_score: float = 1.0
    state: AttentionState = AttentionState.UNKNOWN


class FaceDetector:
    FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    def __init__(self, enable_landmarks: bool = True):
        self.face_cascade = cv2.CascadeClassifier(self.FACE_CASCADE_PATH)
        self.eye_cascade = None
        self.mouth_cascade = None
        
        try:
            eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(eye_path)
        except Exception:
            pass
        
        self.enable_landmarks = enable_landmarks
        self.face_landmarks = None
        self.last_detection_time = 0
        self.detection_cooldown = 0.1
        
        self._eye_history: List[float] = []
        self._blink_timestamps: List[float] = []
        self._last_blink_time = time.time()
        self._was_eye_detected = False

    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        if time.time() - self.last_detection_time < self.detection_cooldown:
            return self.face_landmarks
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            self.face_landmarks = None
            self._was_eye_detected = False
            return None
        
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        landmarks = {
            "face_bbox": (x, y, w, h),
            "face_center": (x + w//2, y + h//2),
            "face_size": w * h,
        }
        
        if self.eye_cascade:
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
            landmarks["eyes"] = [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in eyes]
            
            if len(eyes) >= 2:
                self._was_eye_detected = True
                eye_midpoints = [(x + ex + ew//2, y + ey + eh//2) for ex, ey, ew, eh in eyes]
                landmarks["gaze_direction"] = self._estimate_gaze(eye_midpoints, (x, y, w, h))
        
        if self.enable_landmarks:
            landmarks["head_pose"] = self._estimate_head_pose(x, y, w, h, frame)
            landmarks["expression"] = self._analyze_expression(face_roi)
        
        self.face_landmarks = landmarks
        self.last_detection_time = time.time()
        return landmarks

    def _estimate_gaze(self, eye_midpoints: List[Tuple], face_bbox: Tuple) -> Dict[str, float]:
        if len(eye_midpoints) < 2:
            return {"horizontal": 0.0, "vertical": 0.0, "score": 0.0}
        
        fx, fy, fw, fh = face_bbox
        left_eye, right_eye = eye_midpoints[:2]
        
        eye_distance = abs(left_eye[0] - right_eye[0])
        if eye_distance == 0:
            eye_distance = 1
        
        gaze_horizontal = ((left_eye[0] + right_eye[0]) / 2 - fx - fw/2) / (fw/2)
        gaze_vertical = ((left_eye[1] + right_eye[1]) / 2 - fy - fh/2) / (fh/2)
        
        gaze_score = 1.0 - min(1.0, abs(gaze_horizontal) + abs(gaze_vertical))
        
        return {
            "horizontal": np.clip(gaze_horizontal, -1, 1),
            "vertical": np.clip(gaze_vertical, -1, 1),
            "score": max(0, gaze_score),
        }

    def _estimate_head_pose(self, x: int, y: int, w: int, h: int, 
                           frame: np.ndarray) -> Dict[str, float]:
        center_x, center_y = x + w//2, y + h//2
        frame_center_x, frame_center_y = frame.shape[1]//2, frame.shape[0]//2
        
        yaw = (center_x - frame_center_x) / (frame.shape[1]//2)
        pitch = (center_y - frame_center_y) / (frame.shape[0]//2)
        
        face_aspect = w / h if h > 0 else 1.0
        
        yaw = np.clip(yaw, -1, 1)
        pitch = np.clip(pitch, -1, 1)
        
        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": 0.0,
            "face_aspect": face_aspect,
            "depth_estimation": h / 480.0,
        }

    def _analyze_expression(self, face_roi: np.ndarray) -> Dict[str, float]:
        mean_intensity = np.mean(face_roi)
        std_intensity = np.std(face_roi)
        
        upper_half = face_roi[:face_roi.shape[0]//2, :]
        lower_half = face_roi[face_roi.shape[0]//2:, :]
        
        upper_mean = np.mean(upper_half)
        lower_mean = np.mean(lower_half)
        
        smile_indicator = (lower_mean - upper_mean) / 255.0
        
        return {
            "intensity_mean": mean_intensity / 255.0,
            "intensity_std": std_intensity / 128.0,
            "smile_indicator": np.clip(smile_indicator, -1, 1),
            "neutral_probability": 0.5 + (1.0 - abs(smile_indicator)) * 0.3,
        }

    def get_face_center(self) -> Optional[Tuple[int, int]]:
        if self.face_landmarks and "face_center" in self.face_landmarks:
            return self.face_landmarks["face_center"]
        return None


class AttentionTracker:
    def __init__(self, face_detector: FaceDetector = None):
        self.face_detector = face_detector or FaceDetector()
        self.current_state = AttentionState.UNKNOWN
        self.state_duration = 0.0
        self.state_start_time = time.time()
        self.focus_history: List[Tuple[float, AttentionState]] = []
        
        self.thresholds = {
            "gaze_score_min": 0.5,
            "gaze_horizontal_max": 0.4,
            "gaze_vertical_max": 0.35,
            "head_yaw_max": 0.5,
            "head_pitch_max": 0.4,
            "face_missing_threshold": 2.0,
            "distraction_threshold": 3.0,
        }
        
        self._last_face_detection = time.time()
        self._consecutive_misses = 0
        self._attention_scores: List[float] = []
        self._session_metrics = {
            "total_time": 0.0,
            "focused_time": 0.0,
            "distracted_time": 0.0,
            "away_time": 0.0,
            "attention_switches": 0,
        }

    def update(self, frame: np.ndarray) -> GazeMetrics:
        current_time = time.time()
        frame_duration = current_time - self.state_start_time
        
        self._session_metrics["total_time"] += frame_duration
        self.state_duration += frame_duration
        self.state_start_time = current_time
        
        landmarks = self.face_detector.detect(frame)
        
        metrics = GazeMetrics()
        
        if landmarks is None:
            self._consecutive_misses += 1
            time_missing = self._consecutive_misses * self.face_detector.detection_cooldown
            
            if time_missing > self.thresholds["face_missing_threshold"]:
                self._update_state(AttentionState.AWAY)
                metrics.state = AttentionState.AWAY
                metrics.attention_score = 0.0
                self._session_metrics["away_time"] += frame_duration
        else:
            self._consecutive_misses = 0
            self._last_face_detection = current_time
            
            if "gaze_direction" in landmarks:
                gaze = landmarks["gaze_direction"]
                metrics.eye_aspect_ratio = gaze["score"]
                
                is_gazing = (
                    gaze["score"] >= self.thresholds["gaze_score_min"] and
                    abs(gaze["horizontal"]) <= self.thresholds["gaze_horizontal_max"] and
                    abs(gaze["vertical"]) <= self.thresholds["gaze_vertical_max"]
                )
                
                head_pose = landmarks.get("head_pose", {})
                is_facing_forward = (
                    abs(head_pose.get("yaw", 0)) <= self.thresholds["head_yaw_max"] and
                    abs(head_pose.get("pitch", 0)) <= self.thresholds["head_pitch_max"]
                )
                
                if is_gazing and is_facing_forward:
                    self._update_state(AttentionState.FOCUSED)
                    metrics.state = AttentionState.FOCUSED
                    metrics.attention_score = min(1.0, gaze["score"] + 0.2)
                    self._session_metrics["focused_time"] += frame_duration
                else:
                    self._update_state(AttentionState.DISTRACTED)
                    metrics.state = AttentionState.DISTRACTED
                    gaze_factor = gaze["score"]
                    head_factor = 1.0 - (abs(head_pose.get("yaw", 0)) + abs(head_pose.get("pitch", 0))) / 2
                    metrics.attention_score = max(0.0, min(1.0, (gaze_factor + head_factor) / 2))
                    self._session_metrics["distracted_time"] += frame_duration
                
                self._attention_scores.append(metrics.attention_score)
                if len(self._attention_scores) > 100:
                    self._attention_scores.pop(0)
            else:
                self._update_state(AttentionState.UNKNOWN)
                metrics.state = AttentionState.UNKNOWN
                metrics.attention_score = 0.5
        
        metrics.head_yaw = landmarks.get("head_pose", {}).get("yaw", 0.0) if landmarks else 0.0
        metrics.head_pitch = landmarks.get("head_pose", {}).get("pitch", 0.0) if landmarks else 0.0
        
        return metrics

    def _update_state(self, new_state: AttentionState):
        if new_state != self.current_state:
            if self.current_state != AttentionState.UNKNOWN:
                self._session_metrics["attention_switches"] += 1
            self.focus_history.append((time.time(), new_state))
            self.current_state = new_state
            self.state_duration = 0.0

    def get_current_state(self) -> AttentionState:
        return self.current_state

    def get_attention_score(self) -> float:
        if not self._attention_scores:
            return 1.0
        return np.mean(self._attention_scores[-30:])

    def get_session_metrics(self) -> Dict:
        total = max(0.1, self._session_metrics["total_time"])
        return {
            "total_time_seconds": self._session_metrics["total_time"],
            "focused_time_seconds": self._session_metrics["focused_time"],
            "distracted_time_seconds": self._session_metrics["distracted_time"],
            "away_time_seconds": self._session_metrics["away_time"],
            "attention_switches": self._session_metrics["attention_switches"],
            "focus_percentage": (self._session_metrics["focused_time"] / total) * 100,
            "attention_score": self.get_attention_score(),
            "current_state": self.current_state.value,
        }

    def is_user_attending(self) -> bool:
        return self.current_state == AttentionState.FOCUSED

    def is_user_away(self) -> bool:
        return self.current_state == AttentionState.AWAY

    def get_posture_attention_factor(self) -> float:
        attention = self.get_attention_score()
        
        if self.current_state == AttentionState.FOCUSED:
            return 1.0 + attention * 0.1
        elif self.current_state == AttentionState.DISTRACTED:
            return 0.85 + attention * 0.1
        elif self.current_state == AttentionState.AWAY:
            return 0.6
        return 0.7

    def reset(self):
        self.current_state = AttentionState.UNKNOWN
        self.state_duration = 0.0
        self.state_start_time = time.time()
        self.focus_history = []
        self._attention_scores = []
        self._session_metrics = {
            "total_time": 0.0,
            "focused_time": 0.0,
            "distracted_time": 0.0,
            "away_time": 0.0,
            "attention_switches": 0,
        }


class CombinedPostureAttentionAnalyzer:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.attention_tracker = AttentionTracker(self.face_detector)
        self._combined_history: List[Dict] = []

    def analyze(self, frame: np.ndarray, posture_score: float, 
                posture_label: str) -> Dict:
        attention_metrics = self.attention_tracker.update(frame)
        attention_factor = self.attention_tracker.get_posture_attention_factor()
        
        combined_score = posture_score * attention_factor
        
        analysis = {
            "posture_score": posture_score,
            "attention_score": attention_metrics.attention_score,
            "attention_state": attention_metrics.state.value,
            "combined_score": combined_score,
            "is_attending": self.attention_tracker.is_user_attending(),
            "is_away": self.attention_tracker.is_user_away(),
            "attention_factor": attention_factor,
            "gaze_horizontal": attention_metrics.head_yaw,
            "gaze_vertical": attention_metrics.head_pitch,
        }
        
        self._combined_history.append(analysis)
        if len(self._combined_history) > 1000:
            self._combined_history.pop(0)
        
        return analysis

    def get_session_summary(self) -> Dict:
        posture_attention = self.attention_tracker.get_session_metrics()
        
        if not self._combined_history:
            return {"error": "No data available"}
        
        recent_scores = [c["combined_score"] for c in self._combined_history[-100:]]
        recent_attention = [c["attention_score"] for c in self._combined_history[-100:]]
        
        return {
            "session_metrics": posture_attention,
            "avg_combined_score": np.mean(recent_scores),
            "avg_posture_score": np.mean([c["posture_score"] for c in self._combined_history[-100:]]),
            "avg_attention_score": np.mean(recent_attention),
            "away_episodes": sum(1 for c in self._combined_history[-100:] if c["is_away"]),
            "focus_episodes": sum(1 for c in self._combined_history[-100:] if c["is_attending"]),
        }

    def draw_attention_overlay(self, frame: np.ndarray) -> np.ndarray:
        output = frame.copy()
        
        face_center = self.face_detector.get_face_center()
        if face_center:
            state = self.current_state if hasattr(self, 'current_state') else self.attention_tracker.current_state
            
            if state == AttentionState.FOCUSED:
                color = (0, 255, 0)
                text = "FOCUSED"
            elif state == AttentionState.DISTRACTED:
                color = (0, 255, 255)
                text = "DISTRACTED"
            elif state == AttentionState.AWAY:
                color = (0, 0, 255)
                text = "AWAY"
            else:
                color = (128, 128, 128)
                text = "UNKNOWN"
            
            cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            attention = self.attention_tracker.get_attention_score()
            cv2.putText(output, f"Attention: {attention:.0%}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return output


def create_attention_tracker() -> AttentionTracker:
    return AttentionTracker()


def create_combined_analyzer() -> CombinedPostureAttentionAnalyzer:
    return CombinedPostureAttentionAnalyzer()


if __name__ == "__main__":
    print("Testing Face Detection & Attention Tracking...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera, using test mode")
    else:
        tracker = create_attention_tracker()
        
        print("Starting attention tracking... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            metrics = tracker.update(frame)
            print(f"State: {metrics.state.value}, Score: {metrics.attention_score:.2f}")
            
            cv2.imshow("Attention Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\nSession Summary:")
        print(tracker.get_session_metrics())
