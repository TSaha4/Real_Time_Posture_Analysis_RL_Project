import argparse
import sys
import time
import cv2
import numpy as np
import json
import uuid
from datetime import datetime

from config import config
from posture_module import PostureLabel, RuleBasedClassifier, AdaptiveThresholdClassifier, encode_label, PostureClassifier
from environment import PostureEnvironment, RuleBasedEnvironment, Action, PostureState
from feedback import VisualFeedback, FeedbackLevel
from utils import MetricsLogger, SessionLogger, RateCalculator, setup_directories, validate_config
from algorithm_selector import AlgorithmSelector, create_algorithm_selector


class PostureSystem:
    def __init__(self, algorithm: str = "ppo", camera_id: int = 0, 
                 enable_audio: bool = False, enable_multicamera: bool = False,
                 enable_online_learning: bool = False, skip_calibration: bool = False,
                 enable_attention: bool = False, enable_hands: bool = False,
                 enable_dashboard: bool = False, use_enhanced_training: bool = False,
                 enable_auto_switch: bool = True):
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
        self.enable_auto_switch = enable_auto_switch
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

        # NEW: Suggestion effectiveness tracking
        self.suggestion_effectiveness = {}  # suggestion -> [(score_before, score_after), ...]
        self.last_suggestion_time = 0
        self.last_score_before_suggestion = 0.8

        # NEW: RL action tracking
        self.last_action_time = 0
        self.last_alert_time = 0
        self.last_alert_action = 0
        self.last_score_before_alert = 0.0
        self.first_decision = True

        # Current features for RL state
        self.current_features = {}

        # NEW: Continuous recalibration tracking
        self.good_posture_streak = 0
        self.recalibration_samples = []
        self.recalibration_threshold = 1800  # ~60 sec at 30fps

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
        # Load agent for ppo/dqn (not for rule or auto - use ppo as default)
        if self.algorithm in ["ppo", "dqn"]:
            print(f"Loading {self.algorithm.upper()} agent...")
            self.rl_agent = self._load_agent()
            if self.rl_agent is None:
                print(f"No trained {self.algorithm.upper()} agent found. Please train first.")
                return False
        elif self.algorithm == "auto":
            # For auto mode, load both PPO and DQN agents
            print("Loading PPO agent for auto-switch mode...")
            from rl_ppo_agent import PPOPPOAgent
            self.rl_agent = PPOPPOAgent(state_size=config.rl.state_size, action_size=config.rl.action_size)
            if not self.rl_agent.load(f"{config.system.model_dir}/ppo_final.pth"):
                print("No trained PPO agent found. Please train first.")
                return False
            # Note: DQN will be loaded when needed
            
        print("Initializing pose detector...")
        from pose_module import PoseDetector, PoseAnalyzer
        self.pose_detector = PoseDetector()
        self.pose_analyzer = PoseAnalyzer()

        # Use RuleBasedClassifier - more robust without baseline
        self.classifier = RuleBasedClassifier()
        
        # Initialize auto-switching algorithm selector
        if self.enable_auto_switch:
            print("Initializing auto-switching algorithm selector...")
            self.algorithm_selector = create_algorithm_selector(
                evaluation_interval=30.0,
                switch_threshold=0.10,
                min_alerts=5
            )
        else:
            self.algorithm_selector = None

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
            agent = PPOPPOAgent(state_size=config.rl.state_size, action_size=config.rl.action_size)
            if agent.load(f"{config.system.model_dir}/ppo_final.pth"):
                return agent
        elif self.algorithm == "dqn":
            from rl_agent import DQNAgent
            agent = DQNAgent(state_size=config.rl.state_size, action_size=config.rl.action_size)
            if agent.load(f"{config.system.model_dir}/dqn_final.pth"):
                return agent
        return None

    def run_calibration(self) -> bool:
        print(f"\n{'='*60}\nCALIBRATION PHASE\n{'='*60}")
        print(f"Position yourself in good posture. Keep still for ~{self.calibration_target} frames.")
        print("Press 'c' to capture and continue, 'q' to quit\n")

        # Reset EMA to ensure clean baseline computation
        self.pose_analyzer.reset_ema()

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
            values = [f[key] for f in features_list]
            # Use median for robustness against outliers
            median_val = np.median(values)
            std_val = np.std(values)
            # Reject outliers beyond 2 std dev
            filtered = [v for v in values if abs(v - median_val) <= 2 * std_val]
            baseline[key] = np.mean(filtered) if filtered else median_val
        return baseline
    
    def _get_default_baseline(self) -> dict:
        return {
            "neck_angle": 45,       # Good posture: 40-50 (nose slightly above shoulders)
            "shoulder_diff": 5,      # Good: level shoulders (<10)
            "spine_inclination": 20,   # Good: shoulders well above hips (20-30)
            "forward_head_y": 20,   # Good: nose above shoulders (15-25)
            "head_tilt": 3,         # Good: level head (<5)
            "elbow_asymmetry": 5,     # Good: relaxed arms
            "shoulder_alignment": 10, # Good: aligned shoulders
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
                # Store features for RL state
                if features:
                    self.current_features = features.copy()
                
                if features and self.baseline:
                    features = self._adjust_features_relative_to_baseline(features)
                    new_label, new_score = self.classifier.classify(features)
                    if new_label != self.current_label:
                        # NEW: Track suggestion effectiveness when label changes
                        if self.current_suggestion and self.current_suggestion != "Calibration needed":
                            self._track_suggestion_effectiveness(self.current_suggestion, self.last_score_before_suggestion, new_score)
                        self.current_label = new_label
                        self.current_suggestion = self.classifier.get_suggestion(self.current_label, features)
                        self.last_suggestion_time = self.current_time
                        self.last_score_before_suggestion = self.current_score
                    self.current_score = new_score

                    # NEW: Continuous recalibration - track good posture streaks
                    self._update_recalibration(new_score, keypoints)
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
                attention_score = combined_metrics.attention_score
            else:
                attention_score = 1.0
            
            if self.dashboard_exporter:
                self.dashboard_exporter.record_frame(
                    self.current_score, 
                    self.current_label.value,
                    attention_score
                )
            
            action = 0
            if self.current_time - self.last_decision_time >= config.system.decision_interval:
                self.last_decision_time = self.current_time
                
                # Get active algorithm (auto-switch or fixed)
                active_algo = self._get_active_algorithm()
                
                if active_algo == "rule":
                    action = self._get_rule_action()
                elif self.is_calibrated:
                    action = self._get_rl_action()
                    if self.enable_online_learning and self.rl_agent:
                        self._online_learning_update(action)
                
                # Update algorithm selector with action and score
                if self.algorithm_selector:
                    self.algorithm_selector.record_alert(action, self.current_score)
                    self.algorithm_selector.record_posture_change(self.current_score)
                    
                    # Check if should switch algorithms
                    if self.algorithm_selector.should_switch():
                        new_algo = self.algorithm_selector.switch_algorithm()
                        # Reload agent for new algorithm if needed
                        if new_algo != "rule" and new_algo != self.algorithm:
                            self._load_agent_for_algorithm(new_algo)
                            self.algorithm = new_algo
                            print(f"[Auto-Switch] Switched to {new_algo.upper()}")
                
# Track action timing for RL state
                self.last_action_time = self.current_time
                
                if self.rl_agent:
                    self.current_action_name = self.rl_agent.get_action_name(action)
                else:
                    self.current_action_name = Action(action).name

                # Track alert for correction effectiveness
                if action != 0:
                    self.last_alert_time = self.current_time
                    self.last_alert_action = action
                    self.last_score_before_alert = self.current_score

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
        """Online learning: collect experiences and periodically train the agent"""
        if not self.current_state or not self.rl_agent:
            return
        
        self.online_learning_counter += 1
        
        # Collect experience: state, action, reward
        # We'll compute reward based on posture improvement
        reward = self._compute_online_reward(action)
        
        exp = {
            "state": self.current_state.to_array(),
            "action": action,
            "reward": reward,
            "posture_label": self.current_label.value,
            "score": self.current_score,
        }
        
        self.online_learning_buffer.append(exp)
        
        # Keep buffer at max size
        max_buffer = config.online_learning.batch_size * 4
        if len(self.online_learning_buffer) > max_buffer:
            self.online_learning_buffer = self.online_learning_buffer[-max_buffer:]
        
        # Train periodically when buffer has enough samples
        train_interval = config.online_learning.update_frequency
        if self.online_learning_counter > 0 and self.online_learning_counter % train_interval == 0:
            self._perform_online_training()
    
    def _compute_online_reward(self, action: int) -> float:
        """Compute reward for online learning based on posture and action"""
        if self.current_label == PostureLabel.GOOD:
            if action == 0:  # No feedback - good posture maintained
                return 0.3
            else:  # Alert when already good - unnecessary
                return -0.1
        else:  # Bad posture
            if action == 0:  # No feedback - ignoring bad posture
                return -0.2
            else:  # Alert issued
                return 0.1  # Small positive for trying to correct
    
    def _perform_online_training(self):
        """Perform online training using collected experiences"""
        if not self.rl_agent or len(self.online_learning_buffer) < config.online_learning.min_experiences:
            return
        
        # Use recent experiences for training
        buffer_size = min(len(self.online_learning_buffer), config.online_learning.batch_size * 2)
        recent_buffer = self.online_learning_buffer[-buffer_size:]
        
        # Sample random batch
        import random
        batch_size = min(config.online_learning.batch_size, len(recent_buffer))
        batch = random.sample(recent_buffer, batch_size)
        
        total_loss = 0.0
        num_updates = 0
        
        for exp in batch:
            state = exp["state"]
            action = exp["action"]
            reward = exp["reward"]
            
            # For DQN: need next_state and done
            # Use current state as proxy (simple approach)
            next_state = state  # Simplified
            done = False
            
            # Try to update based on agent type
            agent_type = self._get_active_algorithm()
            
            if agent_type == "dqn":
                # DQN uses update(state, action, reward, next_state, done)
                loss = self.rl_agent.update(state, action, reward, next_state, done)
                if loss is not None:
                    total_loss += loss
                    num_updates += 1
            
            elif agent_type == "ppo":
                # PPO uses get_action (which records to trajectory) and then update()
                # For online learning, we need to do a mini-update
                # Get action to record in trajectory
                self.rl_agent.get_action(state, training=True)
                self.rl_agent.record_step(reward, done)
        
        # For PPO, we need to actually update the policy
        if agent_type == "ppo" and len(self.rl_agent.trajectory) >= batch_size:
            losses = self.rl_agent.update(next_value=0.0)
            if losses and losses.get("loss", 0) > 0:
                total_loss += losses["loss"]
                num_updates += 1
        
        if num_updates > 0:
            avg_loss = total_loss / num_updates
            if self.online_learning_counter % 500 == 0:
                print(f"Online learning: updated {num_updates} times, avg loss: {avg_loss:.4f}")

    def _get_active_algorithm(self) -> str:
        """Get the currently active algorithm - either from auto-switch or fixed setting"""
        if self.algorithm_selector and self.enable_auto_switch:
            return self.algorithm_selector.get_active_algorithm()
        # Handle "auto" as default to ppo when no selector
        if self.algorithm == "auto":
            return "ppo"
        return self.algorithm
    
    def _load_agent_for_algorithm(self, algo: str):
        """Load RL agent for specified algorithm"""
        if algo == "ppo":
            from rl_ppo_agent import PPOPPOAgent
            self.rl_agent = PPOPPOAgent(state_size=config.rl.state_size, action_size=config.rl.action_size)
            self.rl_agent.load(f"{config.system.model_dir}/ppo_final.pth")
        elif algo == "dqn":
            from rl_agent import DQNAgent
            self.rl_agent = DQNAgent(state_size=config.rl.state_size, action_size=config.rl.action_size)
            self.rl_agent.load(f"{config.system.model_dir}/dqn_final.pth")

    def _get_rl_action(self) -> int:
        # Build state in SAME format as training (simulation_enhanced.py line 308)
        # This matches the 18-dim state that PPO/DQN was trained on
        
        # Track local state
        if not hasattr(self, '_rl_consecutive_alerts'):
            self._rl_consecutive_alerts = 0
        if not hasattr(self, '_rl_alerts_this_episode'):
            self._rl_alerts_this_episode = 0
        if not hasattr(self, '_rl_corrections_this_episode'):
            self._rl_corrections_this_episode = 0
        if not hasattr(self, '_rl_correction_streak'):
            self._rl_correction_streak = 0
        if not hasattr(self, '_rl_max_streak'):
            self._rl_max_streak = 0
        if not hasattr(self, '_rl_user_fatigue'):
            self._rl_user_fatigue = 0.0
        if not hasattr(self, '_rl_motivation'):
            self._rl_motivation = 0.7
        if not hasattr(self, '_rl_frustration'):
            self._rl_frustration = 0.0
        
        # Track consecutive alerts
        if self.last_alert_action > 0:
            self._rl_consecutive_alerts += 1
            self._rl_alerts_this_episode += 1
        else:
            self._rl_consecutive_alerts = 0
        
        # Track corrections (when posture improves after alert)
        if self.last_alert_time > 0:
            if self.current_label == PostureLabel.GOOD or self.current_score > self.last_score_before_alert + 0.15:
                self._rl_corrections_this_episode += 1
                self._rl_correction_streak += 1
                self._rl_max_streak = max(self._rl_max_streak, self._rl_correction_streak)
            else:
                self._rl_correction_streak = 0
        
        # Track user fatigue (increases with alerts)
        self._rl_user_fatigue = min(1.0, self._rl_alerts_this_episode / 20.0)
        
        # Compute correction rate
        correction_rate = 0.0
        if self._rl_alerts_this_episode > 0:
            correction_rate = self._rl_corrections_this_episode / self._rl_alerts_this_episode
        
        # Build state array (18 dims to match training)
        is_good = 1.0 if self.current_label == PostureLabel.GOOD else 0.0
        badness = 1.0 - self.current_score
        
        state = np.array([
            # Basic features (6) - matching training format
            float(encode_label(self.current_label)),
            self.current_score,
            self._rl_consecutive_alerts / 5.0,
            self._rl_user_fatigue,
            self._rl_motivation,
            self._rl_frustration,
            # Derived features (6)
            is_good,
            badness,
            self._rl_alerts_this_episode / 20.0,
            correction_rate,
            self._rl_correction_streak / 10.0,
            self._rl_max_streak / 20.0,
            # User profile features (6)
            0.7,  # compliance_rate - default
            0.8,  # attention_span
            0.3,  # stubbornness
            0.0,  # session_time normalized
            0.0,  # correction_ability
            0.5,  # alert_sensitivity
        ], dtype=np.float32)
        
        # Get action from RL agent
        action = self.rl_agent.get_action(state)
        
        # Reset episode tracking if we just started fresh decision cycle
        if self._rl_alerts_this_episode > 15:
            self._rl_alerts_this_episode = 0
            self._rl_corrections_this_episode = 0
            self._rl_max_streak = 0
        
        return action

    def _get_rule_action(self) -> int:
        should_alert, action = self.rule_env.should_alert(
            self.current_label, self.current_state.duration_bad_posture if self.current_state else 0, self.current_time)
        return action

    def _update_recalibration(self, new_score: float, keypoints: dict):
        if new_score >= 0.65:
            self.good_posture_streak += 1
        else:
            self.good_posture_streak = 0
        
        if self.good_posture_streak >= self.recalibration_threshold:
            if keypoints:
                self.recalibration_samples.append(keypoints)
            if len(self.recalibration_samples) >= 10:
                new_baseline = self._compute_baseline_from_samples()
                if new_baseline:
                    self.baseline = new_baseline
                    self.classifier.set_baseline(self.baseline)
                    self.recalibration_samples = []
                    self.good_posture_streak = 0

    def _track_suggestion_effectiveness(self, suggestion: str, score_before: float, score_after: float):
        if suggestion not in self.suggestion_effectiveness:
            self.suggestion_effectiveness[suggestion] = []
        self.suggestion_effectiveness[suggestion].append((score_before, score_after))

    def _compute_baseline_from_samples(self) -> dict:
        if not self.recalibration_samples:
            return None
        features_list = []
        for sample in self.recalibration_samples:
            features = self.pose_analyzer.compute_posture_features(sample)
            if features:
                features_list.append(features)
        if not features_list:
            return None
        baseline = {}
        for key in features_list[0].keys():
            values = [f[key] for f in features_list]
            baseline[key] = np.mean(values)
        return baseline

    def _adjust_features_relative_to_baseline(self, features: dict) -> dict:
        # Now properly computes relative features by comparing current features to baseline
        # The RuleBasedClassifier also does comparison, but this is for logging/debugging
        if not features or not self.baseline:
            return features
        
        adjusted = features.copy()
        adjusted['_deviation'] = {
            'neck': abs(features.get('neck_angle', 50) - self.baseline.get('neck_angle', 45)),
            'forward': features.get('forward_head_y', 20) - self.baseline.get('forward_head_y', 20),
            'spine': abs(features.get('spine_inclination', 30) - self.baseline.get('spine_inclination', 25)),
            'shoulder': abs(features.get('shoulder_diff', 5) - self.baseline.get('shoulder_diff', 5)),
        }
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
                enhanced: bool = False, difficulty: str = "medium"):
    from simulation_enhanced import train_ppo, train_dqn
    import random
    random.seed(42)
    
    print(f"Enhanced Training: {algorithm.upper()} agent for {num_episodes} episodes...")
    if algorithm.lower() == "ppo":
        stats = train_ppo(num_episodes=num_episodes, num_users=num_users, 
                        verbose=True, difficulty=difficulty)
    elif algorithm.lower() == "dqn":
        stats = train_dqn(num_episodes=num_episodes, num_users=num_users,
                        verbose=True, difficulty=difficulty)
    else:
        print(f"Unknown algorithm: {algorithm}")
        return None
    print(f"\nTraining complete! Final eval reward: {stats['eval_rewards'][-1] if stats['eval_rewards'] else 0:.2f}")
    return stats


def train_both(num_episodes: int = 500, num_users: int = 5, enhanced: bool = False, 
              difficulty: str = "medium"):
    from simulation_enhanced import train_ppo, train_dqn
    import random
    random.seed(42)
    
    print("=" * 60 + "\nEnhanced Training - PPO agent\n" + "=" * 60)
    ppo_stats = train_ppo(num_episodes=num_episodes, num_users=num_users,
                        verbose=True, difficulty=difficulty)
    
    print("\n" + "=" * 60 + "\nEnhanced Training - DQN agent\n" + "=" * 60)
    dqn_stats = train_dqn(num_episodes=num_episodes, num_users=num_users,
                        verbose=True, difficulty=difficulty)
    print(f"\n{'='*60}\nTraining Complete!\nPPO: {ppo_stats['eval_rewards'][-1] if ppo_stats['eval_rewards'] else 0:.2f}\nDQN: {dqn_stats['eval_rewards'][-1] if dqn_stats['eval_rewards'] else 0:.2f}\n{'='*60}")
    return {"ppo": ppo_stats, "dqn": dqn_stats}


def compare_agents(num_episodes: int = 100, num_users: int = 5):
    from simulation_enhanced import compare_algorithms
    print("=" * 60 + "\nComparing PPO vs DQN vs Rule-Based\n" + "=" * 60)
    results = compare_algorithms(num_episodes=num_episodes,
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
    parser.add_argument("--algorithm", choices=["ppo", "dqn", "rule", "auto"], default="ppo",
                       help="ppo/dqn/rule or 'auto' for automatic algorithm selection")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium",
                       help="Training difficulty level")
    parser.add_argument("--audio", action="store_true", help="Enable audio alerts")
    parser.add_argument("--multi-camera", action="store_true", help="Enable multi-camera support")
    parser.add_argument("--online-learning", action="store_true", help="Enable online learning")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration and use defaults")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio")
    parser.add_argument("--attention", action="store_true", help="Enable attention tracking")
    parser.add_argument("--hands", action="store_true", help="Enable hand tracking")
    parser.add_argument("--dashboard", action="store_true", help="Enable dashboard export")
    parser.add_argument("--enhanced-training", action="store_true", help="Use enhanced training simulator")
    parser.add_argument("--auto-switch", action="store_true", default=True,
                       help="Enable automatic algorithm switching (enabled by default)")
    parser.add_argument("--no-auto-switch", action="store_true",
                       help="Disable automatic algorithm switching")
    args = parser.parse_args()
    
    enable_audio = args.audio and not args.no_audio
    
    if args.mode == "train":
        train_agent(args.algorithm, args.episodes, args.users, args.enhanced_training, args.difficulty)
    elif args.mode == "train-all":
        train_both(args.episodes, args.users, args.enhanced_training, args.difficulty)
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
            use_enhanced_training=args.enhanced_training,
            enable_auto_switch=args.algorithm == "auto" or not args.no_auto_switch
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
            use_enhanced_training=args.enhanced_training,
            enable_auto_switch=args.algorithm == "auto" or not args.no_auto_switch
        )
        if not system.initialize():
            return 1
        
        if args.skip_calibration:
            system.use_default_calibration()
        elif args.algorithm == "auto":
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
