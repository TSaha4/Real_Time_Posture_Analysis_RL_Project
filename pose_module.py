import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, Dict, Tuple, List
import os
import urllib.request
from config import config


MODEL_FILES = {
    "lite": "pose_landmarker_lite.task",
    "full": "pose_landmarker.task",
    "heavy": "pose_landmarker_heavy.task",
}

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker/float16/1/pose_landmarker.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


def download_mediapipe_model(model_type: str = "lite", force: bool = False) -> bool:
    """Download MediaPipe pose model if not present"""
    if model_type not in MODEL_FILES:
        print(f"Unknown model type: {model_type}")
        return False
    
    model_file = MODEL_FILES[model_type]
    
    if os.path.exists(model_file) and not force:
        print(f"Model already exists: {model_file}")
        return True
    
    if model_type not in MODEL_URLS:
        print(f"No URL available for model: {model_type}")
        return False
    
    url = MODEL_URLS[model_type]
    print(f"Downloading {model_type} model from {url}...")
    
    try:
        urllib.request.urlretrieve(url, model_file)
        print(f"Downloaded {model_file} successfully")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False


def download_all_models() -> Dict[str, bool]:
    """Download all available MediaPipe models"""
    results = {}
    for model_type in MODEL_FILES.keys():
        results[model_type] = download_mediapipe_model(model_type)
    return results


def get_available_model() -> str:
    """Returns the first available model file, preferring lite."""
    for model_type in ["lite", "full", "heavy"]:
        model_file = MODEL_FILES[model_type]
        if os.path.exists(model_file):
            return model_type
    return "lite"  # Default fallback (will fail gracefully if missing)


class PoseDetector:
    LANDMARKS = {
        "nose": 0, "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
        "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
        "left_ear": 7, "right_ear": 8, "mouth_left": 9, "mouth_right": 10,
        "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16, "left_pinky": 17, "right_pinky": 18,
        "left_index": 19, "right_index": 20, "left_thumb": 21, "right_thumb": 22,
        "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28, "left_heel": 29, "right_heel": 30,
        "left_foot_index": 31, "right_foot_index": 32,
    }

    KEY_LANDMARKS = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]

    def __init__(self, model_size: str = "lite", enable_holistic: bool = False):
        # Use available model instead of potentially missing ones
        if model_size not in MODEL_FILES or not os.path.exists(MODEL_FILES[model_size]):
            model_size = get_available_model()
        model_file = MODEL_FILES[model_size]
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"MediaPipe model not found: {model_file}. Please download from MediaPipe repository.")
        
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_file),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=config.pose.min_detection_confidence,
            min_pose_presence_confidence=config.pose.min_tracking_confidence,
            min_tracking_confidence=config.pose.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        
        if enable_holistic:
            options.num_poses = 1
            options.output_segmentation_masks = False
            
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.model_size = model_size
        self.enable_holistic = enable_holistic

    def detect(self, frame: np.ndarray, timestamp_ms: int = 0) -> Optional[Dict]:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        timestamp_ms = max(1, timestamp_ms)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        keypoints = self._extract_keypoints(result.pose_landmarks[0])
        normalized = self._normalize_keypoints(keypoints, frame.shape)
        return normalized

    def _extract_keypoints(self, landmarks) -> Dict[str, Tuple[float, float, float]]:
        keypoints = {}
        for name, idx in self.LANDMARKS.items():
            landmark = landmarks[idx]
            keypoints[name] = (landmark.x, landmark.y, landmark.z)
        return keypoints

    def _normalize_keypoints(self, keypoints: Dict, frame_shape: Tuple) -> Dict[str, Dict]:
        h, w = frame_shape[:2]
        normalized = {}
        for name, coords in keypoints.items():
            normalized[name] = {
                "x": coords[0] * w, "y": coords[1] * h, "z": coords[2] * w,
                "norm_x": coords[0], "norm_y": coords[1], "norm_z": coords[2],
            }
        return normalized

    def get_key_landmarks(self, keypoints: Dict) -> Optional[Dict[str, Dict]]:
        result = {}
        for name in self.KEY_LANDMARKS:
            if name in keypoints:
                result[name] = keypoints[name]
            else:
                return None
        return result

    def draw_skeleton(self, frame: np.ndarray, keypoints: Dict, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        skeleton = [
            ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            ("nose", "left_shoulder"), ("nose", "right_shoulder"),
        ]
        for joint1, joint2 in skeleton:
            if joint1 in keypoints and joint2 in keypoints:
                pt1 = (int(keypoints[joint1]["x"]), int(keypoints[joint1]["y"]))
                pt2 = (int(keypoints[joint2]["x"]), int(keypoints[joint2]["y"]))
                cv2.line(frame, pt1, pt2, color, 2)
        for name in self.KEY_LANDMARKS:
            if name in keypoints:
                pt = (int(keypoints[name]["x"]), int(keypoints[name]["y"]))
                cv2.circle(frame, pt, 5, color, -1)
        return frame


class HolisticPoseDetector:
    def __init__(self, model_complexity: int = 1):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_landmarks = None
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None

    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        keypoints = {}
        
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                name = list(PoseDetector.LANDMARKS.keys())[idx] if idx < len(PoseDetector.LANDMARKS) else f"pose_{idx}"
                keypoints[name] = {
                    "x": landmark.x * frame.shape[1],
                    "y": landmark.y * frame.shape[0],
                    "z": landmark.z * frame.shape[1],
                    "norm_x": landmark.x,
                    "norm_y": landmark.y,
                    "norm_z": landmark.z,
                }
            self.face_landmarks = results.face_landmarks
            self.left_hand_landmarks = results.left_hand_landmarks
            self.right_hand_landmarks = results.right_hand_landmarks
        
        return keypoints if keypoints else None

    def draw_skeleton(self, frame: np.ndarray, keypoints: Dict, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        output = PoseDetector.draw_skeleton(self, frame, keypoints, color)
        
        if self.face_landmarks:
            for landmark in self.face_landmarks.landmark[::5]:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(output, (x, y), 2, (255, 200, 0), -1)
        
        if self.left_hand_landmarks:
            for landmark in self.left_hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(output, (x, y), 3, (0, 200, 255), -1)
        
        if self.right_hand_landmarks:
            for landmark in self.right_hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(output, (x, y), 3, (0, 200, 255), -1)
        
        return output

    def get_hand_position(self) -> Dict:
        return {
            "left_hand": self.left_hand_landmarks,
            "right_hand": self.right_hand_landmarks,
        }


class PoseCalibrator:
    def __init__(self, num_frames: int = 30):
        self.num_frames = num_frames
        self.samples: List[Dict] = []
        self.baseline: Optional[Dict] = None

    def add_sample(self, keypoints: Dict) -> bool:
        if not keypoints:
            return False
        self.samples.append(keypoints)
        return len(self.samples) >= self.num_frames

    def compute_baseline(self) -> Dict:
        if not self.samples:
            raise ValueError("No samples collected")
        keys = self.samples[0].keys()
        self.baseline = {}
        for key in keys:
            values = [s[key] for s in self.samples]
            self.baseline[key] = {
                "x": np.mean([v["x"] for v in values]),
                "y": np.mean([v["y"] for v in values]),
                "norm_x": np.mean([v["norm_x"] for v in values]),
                "norm_y": np.mean([v["norm_y"] for v in values]),
            }
        return self.baseline

    def get_progress(self) -> Tuple[int, int]:
        return len(self.samples), self.num_frames

    def reset(self):
        self.samples = []
        self.baseline = None


class PoseAnalyzer:
    def __init__(self):
        self.detector = PoseDetector()
        self.calibrator = PoseCalibrator(config.system.calibration_frames)
        # NEW: Temporal smoothing with EMA
        self.feature_ema = {}
        self.alpha = 0.3  # smoothing factor

    def reset_ema(self):
        """Reset EMA state - call before calibration or batch processing"""
        self.feature_ema = {}

    def compute_angle(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.degrees(np.arctan2(dy, dx))

    def compute_neck_angle(self, nose: Dict, shoulders: Dict) -> float:
        """
        Calculate neck/head forward position as 0-100 score.
        45 = ideal (nose vertically above shoulders)
        Higher = nose forward of shoulders (forward head)
        Lower = nose behind shoulders (pulled back)
        
        For normalized coords (Y increases downward):
        - nose_y < shoulder_y means nose is ABOVE shoulders (good)
        - nose_y = shoulder_y means nose at shoulder level
        - nose_y > shoulder_y means nose below shoulders (bad)
        """
        shoulder_mid_y = (shoulders["left_shoulder"]["y"] + shoulders["right_shoulder"]["y"]) / 2
        nose_y = nose["y"]
        
        # Offset: negative = good (nose above), positive = bad (nose forward/below)
        offset = nose_y - shoulder_mid_y
        
        # Convert to 0-100 scale where ~45 is ideal
        # Typical offset range: -0.2 (good) to +0.2 (bad)
        # Scale: offset=-0.2 -> 20, offset=0 -> 45, offset=+0.2 -> 70
        score = 45 + (offset * 125)
        
        return max(10, min(90, score))

    def compute_shoulder_alignment(self, shoulders: Dict) -> float:
        """
        Calculate shoulder asymmetry as 0-100 score.
        5 = perfectly aligned (ideal)
        Higher = more asymmetric (one shoulder higher than other)
        """
        y_diff = abs(shoulders["left_shoulder"]["y"] - shoulders["right_shoulder"]["y"])
        x_diff = abs(shoulders["left_shoulder"]["x"] - shoulders["right_shoulder"]["x"])
        
        # Combined difference, scale to reasonable range
        # Typical diff: 0.01-0.1 normalized (0-1 range)
        # Scale to 0-100: multiply by 500 for very small diffs
        asymmetry = np.sqrt(y_diff**2 + x_diff**2)
        return min(100, asymmetry * 500)
    
    def compute_shoulder_diff(self, shoulders: Dict) -> float:
        """
        Calculate shoulder height difference as 0-100 score.
        5 = perfectly level (ideal)
        Higher = one shoulder significantly higher
        """
        y_diff = abs(shoulders["left_shoulder"]["y"] - shoulders["right_shoulder"]["y"])
         
        # Typical range: 0.0-0.1 normalized difference
        # Scale to 0-100: multiply by 500
        # 0.01 diff = 5 (good), 0.1 diff = 50 (bad)
        return min(100, y_diff * 500)

    def compute_head_tilt(self, keypoints: Dict) -> float:
        """Compute head tilt using ear positions (0-10 = good, >20 = tilted)"""
        if "left_ear" in keypoints and "right_ear" in keypoints:
            left_ear = keypoints["left_ear"]
            right_ear = keypoints["right_ear"]
            # Use Y difference for tilt
            y_diff = abs(left_ear["y"] - right_ear["y"])
            # Scale to reasonable range
            return min(50, y_diff * 250)
        return 0.0

    def compute_elbow_asymmetry(self, keypoints: Dict) -> float:
        """NEW: Compute arm tenseness using elbow positions"""
        if "left_elbow" in keypoints and "right_elbow" in keypoints:
            left = keypoints["left_elbow"]
            right = keypoints["right_elbow"]
            y_diff = abs(left["y"] - right["y"])
            # Also check if elbows are dropped (tension indicator)
            left_tension = left["y"]  # Higher Y = lower on screen = more dropped
            right_tension = right["y"]
            return y_diff + abs(left_tension - right_tension)
        return 0.0

    def compute_spine_inclination(self, shoulders: Dict, hips: Dict) -> float:
        """
        Calculate spine forward lean as 0-100 score.
        In webcam view (Y increases downward):
        - shoulders at y=0.5, hips at y=0.85
        - shoulders are ABOVE hips (good) -> offset is negative
        - shoulders at same level as hips (bad) -> offset is near 0
        """
        shoulder_mid_y = (shoulders["left_shoulder"]["y"] + shoulders["right_shoulder"]["y"]) / 2
        hip_mid_y = (hips["left_hip"]["y"] + hips["right_hip"]["y"]) / 2
        
        # offset: typically around -0.35 (shoulders well above hips)
        # More negative = more upright (good)
        # Closer to 0 = more slouched (bad)
        offset = shoulder_mid_y - hip_mid_y  # typically -0.4 to -0.2
        
        # Convert to 0-100 where LOWER = better (more upright)
        # offset in range -0.4 to -0.15 (shoulder above hip)
        # Map: -0.4 (very upright) -> 15, -0.2 (neutral) -> 30, -0.1 (slouched) -> 50
        score = 50 + (offset * 150)
        # Clamp: 10-80 range for practical use
        score = max(10, min(80, score))
        
        return max(0, min(100, score))

    def compute_posture_features(self, keypoints: Dict) -> Dict:
        key_landmarks = self.detector.get_key_landmarks(keypoints)
        if not key_landmarks:
            return None

        shoulders = {k: key_landmarks[k] for k in ["left_shoulder", "right_shoulder"]}
        hips = {k: key_landmarks[k] for k in ["left_hip", "right_hip"]}
        nose = key_landmarks["nose"]

        shoulder_mid_y = (shoulders["left_shoulder"]["y"] + shoulders["right_shoulder"]["y"]) / 2
        torso_height = abs(shoulder_mid_y - hips["left_hip"]["y"])

        # Forward head: nose above shoulders (good) vs at/below shoulders (bad)
        nose_to_shoulder = shoulder_mid_y - nose["y"]  # positive = nose above
        if torso_height > 0:
            forward_head_ratio = nose_to_shoulder / torso_height
        else:
            forward_head_ratio = 0
        forward_head_normalized = max(0, 20 - forward_head_ratio * 200)

        # Compute raw features
        raw_features = {
            "neck_angle": self.compute_neck_angle(nose, shoulders),
            "shoulder_diff": self.compute_shoulder_diff(shoulders),
            "shoulder_alignment": self.compute_shoulder_alignment(shoulders) * 100,
            "spine_inclination": self.compute_spine_inclination(shoulders, hips),
            "forward_head_y": forward_head_normalized,
            "head_tilt": self.compute_head_tilt(keypoints),
            "elbow_asymmetry": self.compute_elbow_asymmetry(keypoints),
        }

        # Apply EMA smoothing for temporal stability
        for key, value in raw_features.items():
            if key in self.feature_ema:
                self.feature_ema[key] = self.alpha * value + (1 - self.alpha) * self.feature_ema[key]
            else:
                self.feature_ema[key] = value

        return self.feature_ema.copy()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    analyzer = PoseAnalyzer()
    cv2.namedWindow("Pose Detection", cv2.WINDOW_NORMAL)
    print("Starting pose detection... Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = analyzer.detector.detect(frame)
        if keypoints:
            frame = analyzer.detector.draw_skeleton(frame, keypoints)
            features = analyzer.compute_posture_features(keypoints)
            if features:
                cv2.putText(frame, f"Neck: {features['neck_angle']:.1f}deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No pose detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
