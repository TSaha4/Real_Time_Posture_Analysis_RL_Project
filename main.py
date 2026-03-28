import argparse
import sys
import time
import cv2
import numpy as np
import json

from config import config
from posture_module import PostureLabel, RuleBasedClassifier, encode_label, PostureClassifier
from environment import PostureEnvironment, RuleBasedEnvironment, Action, PostureState
from feedback import VisualFeedback, FeedbackLevel
from utils import MetricsLogger, SessionLogger, RateCalculator, setup_directories, validate_config


class PostureSystem:
    def __init__(self, algorithm: str = "ppo", camera_id: int = 0, 
                 enable_audio: bool = False, enable_multicamera: bool = False,
                 enable_online_learning: bool = False):
        self.algorithm = algorithm.lower()
        self.camera_id = camera_id
        self.enable_audio = enable_audio
        self.enable_multicamera = enable_multicamera
        self.enable_online_learning = enable_online_learning
        self.cap = None
        self.cap_side = None
        self.pose_detector = None
        self.pose_analyzer = None
        self.classifier = PostureClassifier()
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
        self.online_learning_buffer = []
        self.online_learning_counter = 0
        
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
        print(f"Opening camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Cannot open camera {self.camera_id}")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
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
        calibration_frame_count = 0
        frame_timestamp = 0
        start_time = time.time()
        while calibration_frame_count < self.calibration_target:
            if time.time() - start_time > config.system.calibration_timeout:
                print("Calibration timeout")
                return False
            ret, frame = self.cap.read()
            if not ret:
                continue
            keypoints = self.pose_detector.detect(frame, frame_timestamp)
            frame_timestamp += 33
            remaining = self.calibration_target - calibration_frame_count
            if keypoints:
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
                cv2.putText(frame, "No pose detected - please position yourself", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Remaining: {remaining} frames", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow(config.feedback.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            elif key == ord("c") and calibration_frame_count >= 10:
                break
        if calibration_frame_count < 10:
            print("Not enough calibration samples")
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
            return {}
        baseline = {}
        for key in features_list[0].keys():
            baseline[key] = np.mean([f[key] for f in features_list])
        return baseline

    def run(self):
        print(f"\n{'='*60}\nSTARTING REAL-TIME POSTURE MONITORING ({self.algorithm.upper()})\nAudio: {'ON' if self.enable_audio else 'OFF'} | Multi-camera: {'ON' if self.enable_multicamera else 'OFF'} | Online Learning: {'ON' if self.enable_online_learning else 'OFF'}\nPress 'q' to quit\n{'='*60}\n")
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
                    self.current_label, self.current_score = self.classifier.classify(features)
                    self.current_suggestion = self.classifier.get_suggestion(self.current_label, features)
                else:
                    self.current_label, self.current_score = PostureLabel.UNKNOWN, 0.0
                    self.current_suggestion = "Calibration needed"
                frame = self.pose_detector.draw_skeleton(frame, keypoints)
            elif last_keypoints:
                keypoints = last_keypoints
                self.current_label, self.current_score = PostureLabel.UNKNOWN, 0.0
                self.current_suggestion = "No pose detected"
            action = 0
            action_name = "no_feedback"
            if self.current_time - self.last_decision_time >= config.system.decision_interval:
                self.last_decision_time = self.current_time
                if self.algorithm == "rule":
                    action = self._get_rule_action()
                elif self.is_calibrated:
                    action = self._get_rl_action()
                    if self.enable_online_learning and self.rl_agent:
                        self._online_learning_update(action)
                if self.rl_agent:
                    action_name = self.rl_agent.get_action_name(action)
                else:
                    action_name = Action(action).name
                
                if self.enable_audio and action != 0:
                    self._play_audio_alert(action)
            
            frame = self._draw_overlay(frame, action_name)
            if action != 0:
                self.session_logger.log_frame(self.current_score, action, 0)
            cv2.imshow(config.feedback.window_name, frame)
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
            adjusted["neck_angle"] = features.get("neck_angle", 90) - self.baseline.get("neck_angle", 90) + 90
            adjusted["shoulder_diff"] = abs(features.get("shoulder_diff", 0) - self.baseline.get("shoulder_diff", 0))
            adjusted["spine_inclination"] = abs(features.get("spine_inclination", 0) - self.baseline.get("spine_inclination", 0))
        return adjusted

    def _draw_overlay(self, frame: np.ndarray, action_name: str) -> np.ndarray:
        return self.feedback.draw_overlay(frame, self.current_label.value, self.current_score, action_name, {
            "FPS": f"{self.rate_calc.get_fps():.1f}",
            "Frames": f"{len(self.session_logger.data['posture_scores'])}",
            "Alerts": f"{self.session_logger.data['alerts_sent']}",
            "Corrections": f"{self.session_logger.data['corrections_made']}",
        }, suggestion=self.current_suggestion)
    
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
        print("\nSession Summary:")
        print(json.dumps(self.session_logger.get_summary(), indent=2))


def train_agent(algorithm: str = "ppo", num_episodes: int = 500, num_users: int = 5):
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


def train_both(num_episodes: int = 500, num_users: int = 5):
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
    parser.add_argument("--mode", choices=["realtime", "train", "compare", "train-all"], default="realtime")
    parser.add_argument("--algorithm", choices=["ppo", "dqn", "rule"], default="ppo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--audio", action="store_true", help="Enable audio alerts")
    parser.add_argument("--multi-camera", action="store_true", help="Enable multi-camera support")
    parser.add_argument("--online-learning", action="store_true", help="Enable online learning")
    args = parser.parse_args()
    if args.mode == "train":
        train_agent(args.algorithm, args.episodes, args.users)
    elif args.mode == "train-all":
        train_both(args.episodes, args.users)
    elif args.mode == "compare":
        compare_agents(num_episodes=100, num_users=args.users)
    else:
        system = PostureSystem(
            algorithm=args.algorithm, 
            camera_id=args.camera,
            enable_audio=args.audio,
            enable_multicamera=args.multi_camera,
            enable_online_learning=args.online_learning
        )
        if not system.initialize():
            return 1
        if args.algorithm != "rule" and not system.run_calibration():
            return 1
        system.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
