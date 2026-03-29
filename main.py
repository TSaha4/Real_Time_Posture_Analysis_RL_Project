import argparse
import sys
import time
import cv2
import numpy as np
import json
import uuid
from datetime import datetime

from config import config
from posture_module import PostureLabel, RuleBasedClassifier, encode_label, PostureClassifier
from environment import PostureEnvironment, RuleBasedEnvironment, Action, PostureState
from feedback import VisualFeedback, FeedbackLevel
from utils import MetricsLogger, SessionLogger, RateCalculator, setup_directories, validate_config


class PostureSystem:
    def __init__(self, algorithm: str = "ppo", camera_id: int = 0, 
                 enable_audio: bool = False, enable_multicamera: bool = False,
                 enable_online_learning: bool = False, skip_calibration: bool = False,
                 enable_attention: bool = False, enable_hands: bool = False,
                 enable_dashboard: bool = False, use_enhanced_training: bool = False):
        self.algorithm = algorithm.lower()
        self.camera_id = camera_id
        self.enable_audio = enable_audio
        self.enable_multicamera = enable_multicamera
        self.enable_online_learning = enable_online_learning
        self.skip_calibration = skip_calibration
        self.enable_attention = enable_attention
        self.enable_hands = enable_hands
        self.enable_dashboard = enable_dashboard
        self.use_enhanced_training = use_enhanced_training
        self.cap = None
        self.cap_side = None
        self.pose_detector = None
        self.pose_analyzer = None
        self.classifier = RuleBasedClassifier()
        self.feedback = VisualFeedback()
        self.rl_agent = None
        self.rule_env = RuleBasedEnvironment()
        self.current_state = None
        self.last_decision_time = 0
        self.current_time = 0
        self.metrics_logger = MetricsLogger()
        self.session_logger = SessionLogger()
        self.rate_calc = RateCalculator()
        self.is_calibrated = False
        self.baseline = None
        self.calibration_samples = []
        self.calibration_target = config.system.calibration_frames
        self.current_label = PostureLabel.UNKNOWN
        self.current_score = 0.0
        self.current_suggestion = ""
        self.current_action_name = "no_feedback"
        self.online_learning_buffer = []
        self.online_learning_counter = 0
        
        self.unified_analyzer = None
        self.dashboard_exporter = None
        self.current_session_id = None
        
        if self.enable_audio:
            try:
                from audio_alerts import AudioAlertSystem, AlertSound
                self.audio_alerts = AudioAlertSystem(enabled=True)
            except ImportError:
                self.audio_alerts = None
        else:
            self.audio_alerts = None

    def initialize(self):
        setup_directories()
        if not validate_config():
            print("Configuration validation failed")
            return False
        if self.algorithm != "rule":
            print(f"Loading {self.algorithm.upper()} agent...")
            self.rl_agent = self._load_agent()
            if self.rl_agent is None:
                print(f"No trained {self.algorithm.upper()} agent found. Please train first.")
                return False
        print("Initializing pose detector...")
        from pose_module import PoseDetector, PoseAnalyzer
        self.pose_detector = PoseDetector()
        self.pose_analyzer = PoseAnalyzer()
        
        if self.enable_attention or self.enable_hands:
            print("Initializing unified analyzer...")
            from combined_analyzer import create_unified_analyzer
            self.unified_analyzer = create_unified_analyzer(
                enable_attention=self.enable_attention,
                enable_hands=self.enable_hands
            )
        
        if self.enable_dashboard:
            print("Initializing dashboard exporter...")
            from web_dashboard import create_dashboard_exporter
            self.dashboard_exporter = create_dashboard_exporter()
            self.current_session_id = f"session_{uuid.uuid4().hex[:8]}"
            self.dashboard_exporter.start_session(self.current_session_id, "default_user")
        
        print(f"Opening camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Cannot open camera {self.camera_id}")
            return False
        resolutions = [(1920, 1080), (1280, 720), (640, 480), (640, 360)]
        for target_w, target_h in resolutions:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w >= target_w * 0.8:
                print(f"Camera resolution: {actual_w}x{actual_h} (requested {target_w}x{target_h})")
                self.native_width = actual_w
                self.native_height = actual_h
                break
        else:
            self.native_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.native_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera resolution: {self.native_width}x{self.native_height}")
        print(f"System initialized with {self.algorithm.upper()} agent")
        return True

    def _load_agent(self):
        if self.algorithm == "ppo":
            from rl_ppo_agent import PPOPPOAgent
            agent = PPOPPOAgent(state_size=6, action_size=3)
            if agent.load(f"{config.system.model_dir}/ppo_final.pth"):
                return agent
        elif self.algorithm == "dqn":
            from rl_agent import DQNAgent
            agent = DQNAgent(state_size=6, action_size=3)
            if agent.load(f"{config.system.model_dir}/dqn_final.pth"):
                return agent
        return None

    def run_calibration(self) -> bool:
        print(f"\n{'='*60}\nCALIBRATION PHASE\n{'='*60}")
        print(f"Position yourself in good posture. Keep still for ~{self.calibration_target} frames.")
        print("Press 'c' to capture and continue, 'q' to quit\n")
        
        cv2.namedWindow(config.feedback.window_name, cv2.WINDOW_NORMAL)
        self._resize_window_to_fit_screen()
        
        calibration_frame_count = 0
        frame_timestamp = 0
        start_time = time.time()
        last_detection_time = time.time()
        
        while calibration_frame_count < self.calibration_target:
            elapsed = time.time() - start_time
            if elapsed > config.system.calibration_timeout:
                print(f"Calibration timeout ({config.system.calibration_timeout}s)")
                if calibration_frame_count >= config.system.calibration_min_frames:
                    print(f"Using {calibration_frame_count} frames collected so far...")
                    break
                print("Not enough frames collected")
                return False
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            keypoints = self.pose_detector.detect(frame, frame_timestamp)
            frame_timestamp += 33
            
            if keypoints:
                last_detection_time = time.time()
                frame = self.pose_detector.draw_skeleton(frame, keypoints)
                calibration_frame_count += 1
                self.calibration_samples.append(keypoints)
                remaining = self.calibration_target - calibration_frame_count
                progress = calibration_frame_count / self.calibration_target
                
                cv2.putText(frame, f"Calibrating: {calibration_frame_count}/{self.calibration_target}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                bar_width, bar_height = 400, 20
                bar_x, bar_y = (frame.shape[1] - bar_width) // 2, frame.shape[0] - 50
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
            else:
                no_detection_time = time.time() - last_detection_time
                if no_detection_time > 2:
                    cv2.putText(frame, "No pose detected - please position yourself", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Frames collected: {calibration_frame_count}/{self.calibration_target}",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                else:
                    cv2.putText(frame, "Detecting...",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            remaining_time = max(0, int(config.system.calibration_timeout - elapsed))
            cv2.putText(frame, f"Time: {elapsed_min}:{elapsed_sec:02d} | Timeout in: {remaining_time}s",
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            cv2.imshow(config.feedback.window_name, self._prepare_display_frame(frame))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            elif key == ord("c") and calibration_frame_count >= config.system.calibration_min_frames:
                print("Manual calibration triggered by user")
                break
        
        if calibration_frame_count < config.system.calibration_min_frames:
            print(f"Not enough calibration samples (need {config.system.calibration_min_frames}, got {calibration_frame_count})")
            return False
        
        self.baseline = self._compute_baseline()
        self.classifier.set_baseline(self.baseline)
        self.is_calibrated = True
        print(f"\n{'='*60}\nCALIBRATION COMPLETE - Collected {calibration_frame_count} samples\n{'='*60}\n")
        return True

    def _compute_baseline(self) -> dict:
        features_list = []
        for sample in self.calibration_samples:
            features = self.pose_analyzer.compute_posture_features(sample)
            if features:
                features_list.append(features)
        if not features_list:
            return self._get_default_baseline()
        baseline = {}
        for key in features_list[0].keys():
            baseline[key] = np.mean([f[key] for f in features_list])
        return baseline
    
    def _get_default_baseline(self) -> dict:
        return {
            "neck_angle": 90.0,
            "shoulder_diff": 5.0,
            "shoulder_alignment": 10.0,
            "spine_inclination": 0.0,
            "forward_head_y": 15.0,
        }
    
    def use_default_calibration(self):
        print("Using default calibration baseline...")
        self.baseline = self._get_default_baseline()
        self.classifier.set_baseline(self.baseline)
        self.is_calibrated = True

    def _resize_window_to_fit_screen(self):
        target_w = self.native_width
        target_h = self.native_height
        cv2.resizeWindow(config.feedback.window_name, target_w, target_h)
        print(f"Window resized to {target_w}x{target_h}")

    def _prepare_display_frame(self, frame: np.ndarray) -> np.ndarray:
        if not hasattr(self, 'native_width'):
            return frame
        return frame

    def run(self):
        print(f"\n{'='*60}\nSTARTING REAL-TIME POSTURE MONITORING ({self.algorithm.upper()})\n")
        print(f"Audio: {'ON' if self.enable_audio else 'OFF'} | Multi-camera: {'ON' if self.enable_multicamera else 'OFF'} | Online Learning: {'ON' if self.enable_online_learning else 'OFF'}")
        print(f"Attention Tracking: {'ON' if self.enable_attention else 'OFF'} | Hand Tracking: {'ON' if self.enable_hands else 'OFF'} | Dashboard: {'ON' if self.enable_dashboard else 'OFF'}\n{'='*60}\nPress 'q' to quit\n")
        
        last_keypoints = None
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
            self.rate_calc.tick()
            self.current_time = time.time()
            keypoints = self.pose_detector.detect(frame, int(self.current_time * 1000))
            if keypoints:
                last_keypoints = keypoints
                features = self.pose_analyzer.compute_posture_features(keypoints)
                if features and self.baseline:
                    features = self._adjust_features_relative_to_baseline(features)
                    new_label, new_score = self.classifier.classify(features)
                    if new_label != self.current_label:
                        self.current_label = new_label
                        self.current_suggestion = self.classifier.get_suggestion(self.current_label, features)
                    self.current_score = new_score
                else:
                    self.current_label, self.current_score = PostureLabel.UNKNOWN, 0.0
                    self.current_suggestion = "Calibration needed"
                frame = self.pose_detector.draw_skeleton(frame, keypoints)
            elif last_keypoints:
                keypoints = last_keypoints
                self.current_label, self.current_score = PostureLabel.UNKNOWN, 0.0
                self.current_suggestion = "No pose detected"
            
            if self.unified_analyzer:
                combined_metrics = self.unified_analyzer.analyze(frame, self.current_score, self.current_label.value)
                self.current_score = combined_metrics.combined_score
            
            if self.dashboard_exporter:
                self.dashboard_exporter.record_frame(
                    self.current_score, 
                    self.current_label.value,
                    combined_metrics.attention_score if self.unified_analyzer else 1.0
                )
            
            action = 0
            if self.current_time - self.last_decision_time >= config.system.decision_interval:
                self.last_decision_time = self.current_time
                if self.algorithm == "rule":
                    action = self._get_rule_action()
                elif self.is_calibrated:
                    action = self._get_rl_action()
                    if self.enable_online_learning and self.rl_agent:
                        self._online_learning_update(action)
                if self.rl_agent:
                    self.current_action_name = self.rl_agent.get_action_name(action)
                else:
                    self.current_action_name = Action(action).name
                
                if self.enable_audio and action != 0:
                    self._play_audio_alert(action)
            
            frame = self._draw_overlay(frame, self.current_action_name)
            if action != 0:
                self.session_logger.log_frame(self.current_score, action, 0)
                if self.dashboard_exporter:
                    self.dashboard_exporter.record_alert(True)
            
            display_frame = self._prepare_display_frame(frame)
            cv2.imshow(config.feedback.window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self._cleanup()
    
    def _online_learning_update(self, action: int):
        if not self.current_state:
            return
        self.online_learning_counter += 1
        exp = {
            "state": self.current_state.to_array(),
            "action": action,
            "posture_label": self.current_label.value,
            "score": self.current_score,
        }
        self.online_learning_buffer.append(exp)
        if len(self.online_learning_buffer) >= config.online_learning.batch_size:
            self.online_learning_buffer = self.online_learning_buffer[-config.online_learning.batch_size:]

    def _get_rl_action(self) -> int:
        if self.current_state is None:
            self.current_state = PostureState.from_posture(self.current_label, self.current_score)
        self.current_state.posture_score = self.current_score
        self.current_state.posture_label = encode_label(self.current_label)
        return self.rl_agent.get_action(self.current_state.to_array())

    def _get_rule_action(self) -> int:
        should_alert, action = self.rule_env.should_alert(
            self.current_label, self.current_state.duration_bad_posture if self.current_state else 0, self.current_time)
        return action

    def _adjust_features_relative_to_baseline(self, features: dict) -> dict:
        adjusted = features.copy()
        if self.baseline:
            base_features = self.pose_analyzer.compute_posture_features(self.baseline)
            if base_features:
                adjusted["neck_angle"] = features.get("neck_angle", 90) - base_features.get("neck_angle", 90) + 90
                adjusted["shoulder_diff"] = abs(features.get("shoulder_diff", 0) - base_features.get("shoulder_diff", 0))
                adjusted["spine_inclination"] = features.get("spine_inclination", 0) - base_features.get("spine_inclination", 0)
                adjusted["forward_head_y"] = features.get("forward_head_y", 0) - base_features.get("forward_head_y", 0)
        return adjusted

    def _draw_overlay(self, frame: np.ndarray, action_name: str) -> np.ndarray:
        overlay = self.feedback.draw_overlay(frame, self.current_label.value, self.current_score, action_name, {
            "FPS": f"{self.rate_calc.get_fps():.1f}",
            "Frames": f"{len(self.session_logger.data['posture_scores'])}",
            "Alerts": f"{self.session_logger.data['alerts_sent']}",
            "Corrections": f"{self.session_logger.data['corrections_made']}",
        }, suggestion=self.current_suggestion)
        
        if self.unified_analyzer:
            overlay = self._add_combined_overlay(overlay)
        
        return overlay
    
    def _add_combined_overlay(self, frame: np.ndarray) -> np.ndarray:
        if not self.unified_analyzer:
            return frame
        
        metrics = self.unified_analyzer.analyze(frame, self.current_score, self.current_label.value)
        
        num_lines = 1
        if self.enable_attention:
            num_lines += 1
        if self.enable_hands:
            num_lines += 1
        
        x, y = 10, frame.shape[0] - 25 - (num_lines * 18)
        box_h = num_lines * 18 + 10
        box_w = 180
        
        cv2.rectangle(frame, (x-3, y-3), (x + box_w, y + box_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (x-3, y-3), (x + box_w, y + box_h), (60, 60, 60), 1)
        
        y_offset = y + 12
        
        if self.enable_attention:
            att_color = (0, 255, 0) if metrics.is_attending else (0, 255, 255) if not metrics.is_away else (0, 0, 255)
            att_text = f"Att: {metrics.attention_state.upper()[:4]} {metrics.attention_score:.0%}"
            cv2.putText(frame, att_text, (x + 5, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, att_color, 1)
            y_offset += 18
        
        if self.enable_hands:
            hand_text = f"Hands: {metrics.hand_state.upper()[:6]}"
            cv2.putText(frame, hand_text, (x + 5, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            y_offset += 18
        
        cv2.putText(frame, f"Combined: {metrics.combined_score:.0%}", (x + 5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def _play_audio_alert(self, action: int):
        if not self.audio_alerts:
            return
        try:
            from audio_alerts import AlertSound
            if action == 0:
                self.audio_alerts.play_alert(AlertSound.GOOD)
            elif action == 1:
                self.audio_alerts.play_alert(AlertSound.SUBTLE)
            elif action == 2:
                self.audio_alerts.play_alert(AlertSound.STRONG)
        except:
            pass

    def _cleanup(self):
        print("\nCleaning up...")
        self.session_logger.finalize()
        if self.rl_agent:
            self.rl_agent.save(f"{config.system.model_dir}/{self.algorithm}_current.pth")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(json.dumps(self.session_logger.get_summary(), indent=2))
        
        if self.dashboard_exporter:
            self.dashboard_exporter.finalize_session_stats()
            session = self.dashboard_exporter.end_session()
            print("\nDashboard Data Exported")
            print(json.dumps(self.dashboard_exporter.export_for_web(), indent=2)[:500] + "...")
        
        if self.unified_analyzer:
            print("\n" + "="*60)
            print("COMBINED ANALYZER SUMMARY")
            print("="*60)
            print(json.dumps(self.unified_analyzer.get_session_summary(), indent=2))


class HandTrackingSystem:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.hand_tracker = None
    
    def initialize(self):
        from hand_tracker import HandTracker
        self.hand_tracker = HandTracker(use_holistic=True)
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Cannot open camera {self.camera_id}")
            return False
        return True
    
    def run(self):
        print(f"\n{'='*60}\nHAND TRACKING MODE\n{'='*60}\nPress 'q' to quit\n")
        
        cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Tracking", 800, 600)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            hands_data = self.hand_tracker.detect_hands(frame)
            metrics = self.hand_tracker.update_metrics(hands_data)
            
            if hands_data:
                frame = self.hand_tracker.draw_hands(frame, hands_data)
            
            cv2.putText(frame, f"State: {metrics.posture_state.value.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Symmetry: {metrics.symmetry_score:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Tension: {metrics.tension_score:.2f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Typing Intensity: {self.hand_tracker.get_typing_intensity():.2f}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Typing: {'Yes' if self.hand_tracker.is_typing() else 'No'}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self._cleanup()
    
    def _cleanup(self):
        print("\nCleaning up...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("\nSession Summary:")
        print(json.dumps(self.hand_tracker.get_session_metrics(), indent=2))


class AttentionTrackingSystem:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.attention_tracker = None
    
    def initialize(self):
        from attention_tracker import AttentionTracker
        self.attention_tracker = AttentionTracker()
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Cannot open camera {self.camera_id}")
            return False
        return True
    
    def run(self):
        print(f"\n{'='*60}\nATTENTION TRACKING MODE\n{'='*60}\nPress 'q' to quit\n")
        
        cv2.namedWindow("Attention Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Attention Tracking", 800, 600)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            metrics = self.attention_tracker.update(frame)
            
            state = self.attention_tracker.current_state
            if state.value == "focused":
                color = (0, 255, 0)
            elif state.value == "distracted":
                color = (0, 255, 255)
            elif state.value == "away":
                color = (0, 0, 255)
            else:
                color = (128, 128, 128)
            
            cv2.putText(frame, f"State: {state.value.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Attention Score: {metrics.attention_score:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Gaze H: {metrics.head_yaw:.2f} V: {metrics.head_pitch:.2f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            session_metrics = self.attention_tracker.get_session_metrics()
            cv2.putText(frame, f"Focus %: {session_metrics['focus_percentage']:.1f}%", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Attention Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self._cleanup()
    
    def _cleanup(self):
        print("\nCleaning up...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("\nSession Summary:")
        print(json.dumps(self.attention_tracker.get_session_metrics(), indent=2))


def train_agent(algorithm: str = "ppo", num_episodes: int = 500, num_users: int = 5, 
                enhanced: bool = False):
    if enhanced:
        from simulation_enhanced import UnifiedTrainer, TrainingSimulator
        from rl_ppo_agent import PPOPPOAgent
        from rl_agent import DQNAgent
        print(f"Enhanced Training: {algorithm.upper()} agent for {num_episodes} episodes...")
        if algorithm.lower() == "ppo":
            agent = PPOPPOAgent(state_size=6, action_size=3)
        elif algorithm.lower() == "dqn":
            agent = DQNAgent(state_size=6, action_size=3)
        else:
            print(f"Unknown algorithm: {algorithm}")
            return None
        simulator = TrainingSimulator(num_users=num_users, enable_curriculum=True)
        trainer = UnifiedTrainer(agent, simulator, num_episodes=num_episodes, algorithm=algorithm)
        stats = trainer.train()
    else:
        from simulation import train_ppo, train_dqn
        print(f"Training {algorithm.upper()} agent for {num_episodes} episodes...")
        if algorithm.lower() == "ppo":
            stats = train_ppo(num_episodes=num_episodes, num_users=num_users)
        elif algorithm.lower() == "dqn":
            stats = train_dqn(num_episodes=num_episodes, num_users=num_users)
        else:
            print(f"Unknown algorithm: {algorithm}")
            return None
    print(f"\nTraining complete! Final eval reward: {stats['eval_rewards'][-1] if stats['eval_rewards'] else 0:.2f}")
    return stats


def train_both(num_episodes: int = 500, num_users: int = 5, enhanced: bool = False):
    if enhanced:
        from simulation_enhanced import UnifiedTrainer, TrainingSimulator
        from rl_ppo_agent import PPOPPOAgent
        from rl_agent import DQNAgent
        
        print("=" * 60 + "\nEnhanced Training - PPO agent\n" + "=" * 60)
        ppo_agent = PPOPPOAgent(state_size=6, action_size=3)
        ppo_simulator = TrainingSimulator(num_users=num_users, enable_curriculum=True)
        ppo_trainer = UnifiedTrainer(ppo_agent, ppo_simulator, num_episodes=num_episodes, algorithm="ppo")
        ppo_stats = ppo_trainer.train()
        
        print("\n" + "=" * 60 + "\nEnhanced Training - DQN agent\n" + "=" * 60)
        dqn_agent = DQNAgent(state_size=6, action_size=3)
        dqn_simulator = TrainingSimulator(num_users=num_users, enable_curriculum=True)
        dqn_trainer = UnifiedTrainer(dqn_agent, dqn_simulator, num_episodes=num_episodes, algorithm="dqn")
        dqn_stats = dqn_trainer.train()
    else:
        from simulation import train_ppo, train_dqn
        print("=" * 60 + "\nTraining PPO agent...\n" + "=" * 60)
        ppo_stats = train_ppo(num_episodes=num_episodes, num_users=num_users)
        print("\n" + "=" * 60 + "\nTraining DQN agent...\n" + "=" * 60)
        dqn_stats = train_dqn(num_episodes=num_episodes, num_users=num_users)
    print(f"\n{'='*60}\nTraining Complete!\nPPO: {ppo_stats['eval_rewards'][-1] if ppo_stats['eval_rewards'] else 0:.2f}\nDQN: {dqn_stats['eval_rewards'][-1] if dqn_stats['eval_rewards'] else 0:.2f}\n{'='*60}")
    return {"ppo": ppo_stats, "dqn": dqn_stats}


def compare_agents(num_episodes: int = 100, num_users: int = 5):
    from simulation import compare_algorithms
    print("=" * 60 + "\nComparing PPO vs DQN vs Rule-Based\n" + "=" * 60)
    results = compare_algorithms(num_episodes=num_episodes, num_users=num_users,
                                  ppo_path=f"{config.system.model_dir}/ppo_final.pth",
                                  dqn_path=f"{config.system.model_dir}/dqn_final.pth")
    print("\n" + "=" * 60 + "\nCOMPARISON RESULTS\n" + "=" * 60)
    for algo, metrics in results.items():
        print(f"\n{algo}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    best_algo = max(results.items(), key=lambda x: x[1]["avg_episode_reward"])
    print(f"\n{'='*60}\nBest: {best_algo[0]} (Reward: {best_algo[1]['avg_episode_reward']:.4f}, Correction Rate: {best_algo[1]['correction_rate']:.4f})\n{'='*60}")
    return results


def main():
    parser = argparse.ArgumentParser(description="UPRYT - Real-Time Posture Analysis with RL")
    parser.add_argument("--mode", choices=["realtime", "hand", "attention", "combined", "train", "compare", "train-all"], default="realtime")
    parser.add_argument("--algorithm", choices=["ppo", "dqn", "rule"], default="ppo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--audio", action="store_true", help="Enable audio alerts")
    parser.add_argument("--multi-camera", action="store_true", help="Enable multi-camera support")
    parser.add_argument("--online-learning", action="store_true", help="Enable online learning")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration and use defaults")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio")
    parser.add_argument("--attention", action="store_true", help="Enable attention tracking")
    parser.add_argument("--hands", action="store_true", help="Enable hand tracking")
    parser.add_argument("--dashboard", action="store_true", help="Enable dashboard export")
    parser.add_argument("--enhanced-training", action="store_true", help="Use enhanced training simulator")
    args = parser.parse_args()
    
    enable_audio = args.audio and not args.no_audio
    
    if args.mode == "train":
        train_agent(args.algorithm, args.episodes, args.users, args.enhanced_training)
    elif args.mode == "train-all":
        train_both(args.episodes, args.users, args.enhanced_training)
    elif args.mode == "compare":
        compare_agents(num_episodes=100, num_users=args.users)
    elif args.mode == "hand":
        system = HandTrackingSystem(camera_id=args.camera)
        if not system.initialize():
            return 1
        system.run()
    elif args.mode == "attention":
        system = AttentionTrackingSystem(camera_id=args.camera)
        if not system.initialize():
            return 1
        system.run()
    elif args.mode == "combined":
        system = PostureSystem(
            algorithm=args.algorithm, 
            camera_id=args.camera,
            enable_audio=enable_audio,
            enable_multicamera=args.multi_camera,
            enable_online_learning=args.online_learning,
            skip_calibration=args.skip_calibration,
            enable_attention=True,
            enable_hands=True,
            enable_dashboard=True,
            use_enhanced_training=args.enhanced_training
        )
        if not system.initialize():
            return 1
        
        if args.skip_calibration or args.algorithm == "rule":
            system.use_default_calibration()
        else:
            if not system.run_calibration():
                print("\nCalibration failed. You can try:")
                print("  1. Ensure you're visible in the camera")
                print("  2. Use --skip-calibration to use default settings")
                return 1
        
        system.run()
    else:
        system = PostureSystem(
            algorithm=args.algorithm, 
            camera_id=args.camera,
            enable_audio=enable_audio,
            enable_multicamera=args.multi_camera,
            enable_online_learning=args.online_learning,
            skip_calibration=args.skip_calibration,
            enable_attention=args.attention,
            enable_hands=args.hands,
            enable_dashboard=args.dashboard,
            use_enhanced_training=args.enhanced_training
        )
        if not system.initialize():
            return 1
        
        if args.skip_calibration:
            system.use_default_calibration()
        elif args.algorithm != "rule":
            if not system.run_calibration():
                print("\nCalibration failed. You can try:")
                print("  1. Ensure you're visible in the camera")
                print("  2. Use --skip-calibration to use default settings")
                return 1
        else:
            system.use_default_calibration()
        
        system.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
