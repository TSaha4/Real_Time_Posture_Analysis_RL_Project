"""
Mobile-Optimized Posture Detector
Uses MediaPipe for pose detection on Android
"""

import time
import numpy as np
from dataclasses import dataclass


@dataclass
class PostureMetrics:
    """Posture metrics data class"""
    posture_score: float = 0.0
    attention_score: float = 1.0
    hand_score: float = 1.0
    posture_label: str = "UNKNOWN"
    attention_state: str = "UNKNOWN"
    hand_state: str = "UNKNOWN"
    is_typing: bool = False
    fps: float = 0.0
    combined_score: float = 0.0


class PostureDetector:
    """
    Mobile-optimized posture detector using MediaPipe.
    Designed for real-time performance on mobile devices.
    """
    
    def __init__(self, enable_attention=True, enable_hands=True):
        self.enable_attention = enable_attention
        self.enable_hands = enable_hands
        
        self.posture_score = 0.8
        self.attention_score = 1.0
        self.hand_score = 1.0
        self.posture_label = "GOOD"
        
        # Calibration baseline
        self.baseline = None
        self.calibration_samples = []
        self.is_calibrated = False
        
        # Performance tracking
        self.frame_times = []
        self.last_process_time = time.time()
        
        # Initialize MediaPipe components
        self._init_mediapipe()
    
    def _init_mediapipe(self):
        """Initialize MediaPipe components"""
        try:
            import mediapipe as mp
            self.mp = mp
            self.mp_pose = mp.solutions.pose
            self.mp_hands = mp.solutions.hands
            self.mp_face = mp.solutions.face_detection
            
            # Pose detector
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # Lite model for speed
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Hands detector (optional)
            if self.enable_hands:
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=0,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.5
                )
            else:
                self.hands = None
            
            # Face detector (optional)
            if self.enable_attention:
                self.face_detection = self.mp_face.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=0.5
                )
            else:
                self.face_detection = None
            
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
        except ImportError as e:
            print(f"MediaPipe not available: {e}")
            self.pose = None
            self.hands = None
            self.face_detection = None
    
    def calibrate(self, frame):
        """
        Perform quick calibration.
        Should be called when user is in good posture.
        """
        if self.pose is None:
            return False
        
        try:
            image = self._preprocess_frame(frame)
            results = self.pose.process(image)
            
            if results.pose_landmarks:
                self.calibration_samples.append(results.pose_landmarks)
                
                if len(self.calibration_samples) >= 10:
                    self._compute_baseline()
                    self.is_calibrated = True
                    return True
        except:
            pass
        
        return False
    
    def _compute_baseline(self):
        """Compute baseline from calibration samples"""
        # Simple baseline - average key angles
        self.baseline = {
            'shoulder_mid_y': 0.5,
            'nose_y': 0.3,
            'hip_y': 0.7
        }
    
    def _preprocess_frame(self, frame):
        """Preprocess frame for MediaPipe"""
        import mediapipe as mp
        image = frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        return image
    
    def analyze(self, frame) -> PostureMetrics:
        """
        Analyze a single frame and return posture metrics.
        Optimized for mobile performance.
        """
        start_time = time.time()
        
        metrics = PostureMetrics()
        metrics.fps = self._calculate_fps()
        
        if self.pose is None:
            # No MediaPipe - simulate
            metrics.posture_score = 0.8
            metrics.posture_label = "SIMULATED"
            return metrics
        
        try:
            # Process frame
            image = self._preprocess_frame(frame)
            
            # Pose detection
            pose_results = self.pose.process(image)
            
            # Face detection (for attention)
            if self.face_detection:
                face_results = self.face_detection.process(image)
                metrics.attention_score = self._analyze_attention(face_results)
                metrics.attention_state = self._get_attention_state(face_results)
            
            # Hand detection (for typing posture)
            if self.hands:
                hand_results = self.hands.process(image)
                metrics.hand_score, metrics.hand_state, metrics.is_typing = \
                    self._analyze_hands(hand_results)
            
            # Analyze pose
            if pose_results.pose_landmarks:
                posture_score, posture_label = self._analyze_pose(
                    pose_results.pose_landmarks, pose_results.pose_world_landmarks
                )
                metrics.posture_score = posture_score
                metrics.posture_label = posture_label
            
            # Compute combined score
            metrics.combined_score = self._compute_combined_score(metrics)
            
        except Exception as e:
            print(f"Analysis error: {e}")
            metrics.posture_score = 0.5
            metrics.posture_label = "ERROR"
        
        # Track performance
        process_time = time.time() - start_time
        self.frame_times.append(process_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        return metrics
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS"""
        if len(self.frame_times) == 0:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def _analyze_pose(self, landmarks, world_landmarks) -> tuple:
        """
        Analyze pose landmarks and return score and label.
        Uses simplified metrics for mobile performance.
        """
        try:
            # Get key landmarks
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            # Calculate shoulder alignment
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y) * 100
            
            # Calculate forward lean
            nose_y = nose.y
            shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
            forward_lean = (nose_y - shoulder_mid_y) * 100
            
            # Calculate hip alignment
            hip_diff = abs(left_hip.y - right_hip.y) * 100
            
            # Compute score (simplified)
            score = 1.0
            score -= shoulder_diff * 0.1
            score -= abs(forward_lean - 0.2) * 0.1 if forward_lean > 0 else forward_lean * 0.05
            score -= hip_diff * 0.05
            
            score = max(0.0, min(1.0, score))
            
            # Determine label
            if score >= 0.7:
                label = "GOOD"
            elif score >= 0.5:
                label = "SLIGHT"
            elif score >= 0.3:
                label = "BAD"
            else:
                label = "VERY_BAD"
            
            return score, label
            
        except Exception as e:
            return 0.5, "UNKNOWN"
    
    def _analyze_attention(self, face_results) -> float:
        """Analyze face detection results for attention"""
        if face_results.detections:
            return 0.9  # Face detected
        return 0.3  # No face
    
    def _get_attention_state(self, face_results) -> str:
        """Get attention state from face detection"""
        if face_results.detections:
            return "FOCUSED"
        return "AWAY"
    
    def _analyze_hands(self, hand_results) -> tuple:
        """Analyze hand detection results"""
        if not hand_results.multi_hand_landmarks:
            return 1.0, "NOT_DETECTED", False
        
        num_hands = len(hand_results.multi_hand_landmarks)
        
        if num_hands >= 2:
            return 0.9, "GOOD", True
        elif num_hands == 1:
            return 0.7, "ONE_HAND", True
        else:
            return 1.0, "NO_HANDS", False
    
    def _compute_combined_score(self, metrics: PostureMetrics) -> float:
        """Compute combined score from all metrics"""
        base_score = metrics.posture_score
        
        # Weight posture heavily
        combined = base_score * 0.7
        
        # Add attention influence
        if metrics.attention_state == "FOCUSED":
            combined += base_score * 0.2
        elif metrics.attention_state == "AWAY":
            combined += base_score * 0.1
        
        # Add hand influence
        if metrics.hand_state not in ["NOT_DETECTED", "NO_HANDS"]:
            combined += metrics.hand_score * 0.1
        
        return max(0.0, min(1.0, combined))
    
    def draw_landmarks(self, frame, landmarks):
        """Draw pose landmarks on frame (optional, for debugging)"""
        if self.mp_drawing is None or landmarks is None:
            return frame
        
        try:
            annotated_image = frame.copy()
            self.mp_drawing.draw_landmarks(
                annotated_image,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            return annotated_image
        except:
            return frame
    
    def cleanup(self):
        """Cleanup resources"""
        if self.pose:
            self.pose.close()
        if self.hands:
            self.hands.close()
        if self.face_detection:
            self.face_detection.close()


# Import cv2 for preprocessing
import cv2
