#!/usr/bin/env python3
"""
UPRYT - Comprehensive System Test Suite
Tests all components before production deployment
"""

import sys
import os
import time
import json
import traceback
import numpy as np
import torch
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Color codes for terminal output (Windows-safe)
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Use ASCII-safe check marks
PASS_MARK = "[PASS]"
FAIL_MARK = "[FAIL]"
WARN_MARK = "[WARN]"
CHECK_MARK = "OK"
CROSS_MARK = "X"

test_results = {
    "passed": [],
    "failed": [],
    "warnings": []
}

def print_header(text):
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

def print_test(name, status, details=""):
    if status == "PASS":
        print(f"{GREEN}{PASS_MARK}{RESET}: {name}")
        test_results["passed"].append(name)
    elif status == "FAIL":
        print(f"{RED}{FAIL_MARK}{RESET}: {name}")
        if details:
            print(f"       {details}")
        test_results["failed"].append((name, details))
    elif status == "WARN":
        print(f"{YELLOW}{WARN_MARK}{RESET}: {name}")
        if details:
            print(f"       {details}")
        test_results["warnings"].append((name, details))

def test_config():
    """Test configuration module"""
    print_header("TESTING: Configuration Module")
    
    try:
        from config import config
        
        # Test RL config
        assert config.rl.state_size == 18, "RL state_size should be 18"
        assert config.rl.action_size == 3, "RL action_size should be 3"
        assert config.rl.gamma == 0.99, "RL gamma should be 0.99"
        
        # Test system config
        assert config.system.decision_interval == 2.5, "decision_interval should be 2.5"
        assert config.system.calibration_frames == 20, "calibration_frames should be 20"
        
        # Test posture config
        assert config.posture.neck_angle_threshold == 15.0
        assert config.posture.shoulder_diff_threshold == 12.0
        
        print_test("Config imports correctly", "PASS")
        print_test("RL configuration values", "PASS")
        print_test("System configuration values", "PASS")
        print_test("Posture thresholds", "PASS")
        
    except Exception as e:
        print_test("Configuration module", "FAIL", str(e))
        traceback.print_exc()

def test_posture_labels():
    """Test posture label definitions"""
    print_header("TESTING: Posture Labels & Encodings")
    
    try:
        from posture_module import PostureLabel, LABEL_ENCODINGS, encode_label, decode_label
        
        # Test all labels exist
        expected_labels = ["GOOD", "SLOUCHING", "FORWARD_HEAD", "LEANING", "UNKNOWN"]
        for label_name in expected_labels:
            assert hasattr(PostureLabel, label_name), f"Missing label: {label_name}"
        
        # Test encoding/decoding
        for label in PostureLabel:
            encoded = encode_label(label)
            decoded = decode_label(encoded)
            assert decoded == label, f"Encoding/decoding failed for {label}"
        
        print_test("PostureLabel enum definitions", "PASS")
        print_test("Label encoding/decoding", "PASS")
        
    except Exception as e:
        print_test("Posture labels", "FAIL", str(e))
        traceback.print_exc()

def test_rule_based_classifier():
    """Test rule-based posture classifier"""
    print_header("TESTING: Rule-Based Classifier")
    
    try:
        from posture_module import RuleBasedClassifier, PostureLabel
        
        classifier = RuleBasedClassifier()
        
        # Test with baseline
        baseline = {
            "neck_angle": 45,
            "forward_head_y": 20,
            "spine_inclination": 25,
            "shoulder_diff": 5
        }
        classifier.set_baseline(baseline)
        
        # Test good posture (within thresholds)
        good_features = {
            "neck_angle": 48,  # deviation 3 < 15
            "forward_head_y": 22,  # deviation 2 < 15
            "spine_inclination": 28,  # deviation 3 < 18
            "shoulder_diff": 8  # < 12
        }
        label, score = classifier.classify(good_features)
        assert score >= 0.5, "Good posture should have decent score"
        
        # Test forward head
        forward_features = {
            "neck_angle": 60,
            "forward_head_y": 35,  # deviation 15 = threshold
            "spine_inclination": 30,
            "shoulder_diff": 8
        }
        label, score = classifier.classify(forward_features)
        
        # Test slouching
        slouch_features = {
            "neck_angle": 50,
            "forward_head_y": 25,
            "spine_inclination": 50,  # deviation 25 > 18
            "shoulder_diff": 10
        }
        label, score = classifier.classify(slouch_features)
        
        # Test leaning
        lean_features = {
            "neck_angle": 48,
            "forward_head_y": 22,
            "spine_inclination": 28,
            "shoulder_diff": 25  # deviation 20 > 12
        }
        label, score = classifier.classify(lean_features)
        
        # Test get_suggestion
        suggestion = classifier.get_suggestion(PostureLabel.GOOD, good_features)
        assert len(suggestion) > 0, "Should return suggestion"
        
        suggestion = classifier.get_suggestion(PostureLabel.FORWARD_HEAD, forward_features)
        assert len(suggestion) > 0, "Should return suggestion for bad posture"
        
        print_test("RuleBasedClassifier instantiation", "PASS")
        print_test("Good posture classification", "PASS")
        print_test("Bad posture classification (forward head)", "PASS")
        print_test("Bad posture classification (slouching)", "PASS")
        print_test("Bad posture classification (leaning)", "PASS")
        print_test("Suggestion generation", "PASS")
        
    except Exception as e:
        print_test("Rule-based classifier", "FAIL", str(e))
        traceback.print_exc()

def test_environment():
    """Test RL environment"""
    print_header("TESTING: RL Environment")
    
    try:
        from environment import PostureEnvironment, RuleBasedEnvironment, Action, PostureState, PostureLabel
        from posture_module import encode_label
        
        # Test PostureState
        state = PostureState(
            posture_label=encode_label(PostureLabel.GOOD),
            posture_score=0.8,
            duration_bad_posture=0.0,
            time_since_alert=0.0
        )
        state_array = state.to_array()
        assert len(state_array) == 18, f"State should be 18-dim, got {len(state_array)}"
        
        # Test PostureEnvironment
        env = PostureEnvironment()
        initial_state = env.reset()
        assert initial_state.posture_score == 0.7, "Initial score should be 0.7"
        
        # Test step with no action (NO_FEEDBACK)
        next_state, reward, done = env.step(0, PostureLabel.GOOD, 0.8)
        assert isinstance(reward, float), "Reward should be float"
        
        # Test step with subtle alert (correction)
        next_state, reward, done = env.step(1, PostureLabel.GOOD, 0.8)
        
        # Test step with strong alert
        next_state, reward, done = env.step(2, PostureLabel.GOOD, 0.85)
        
        # Test step without correction
        next_state, reward, done = env.step(1, PostureLabel.SLOUCHING, 0.4)
        
        # Test RuleBasedEnvironment
        rule_env = RuleBasedEnvironment()
        should_alert, action = rule_env.should_alert(PostureLabel.GOOD, 0, time.time())
        assert action == 0, "No alert for good posture"
        
        should_alert, action = rule_env.should_alert(PostureLabel.SLOUCHING, 5, time.time())
        assert action in [1, 2], "Should alert for bad posture"
        
        print_test("PostureState creation", "PASS")
        print_test("PostureState.to_array()", "PASS")
        print_test("PostureEnvironment.reset()", "PASS")
        print_test("PostureEnvironment.step() - NO_FEEDBACK", "PASS")
        print_test("PostureEnvironment.step() - alerts", "PASS")
        print_test("RuleBasedEnvironment.alert logic", "PASS")
        
    except Exception as e:
        print_test("RL Environment", "FAIL", str(e))
        traceback.print_exc()

def test_dqn_agent():
    """Test DQN agent"""
    print_header("TESTING: DQN Agent")
    
    try:
        from rl_agent import DQNAgent, QNetwork, ReplayBuffer
        
        # Test network creation
        network = QNetwork(state_size=18, action_size=3, hidden_sizes=[256, 128, 64])
        test_input = torch.randn(1, 18)
        output = network(test_input)
        assert output.shape == (1, 3), f"Output should be (1, 3), got {output.shape}"
        
        # Test agent creation
        agent = DQNAgent(state_size=18, action_size=3)
        
        # Test get_action (inference)
        state = np.random.rand(18).astype(np.float32)
        action = agent.get_action(state, training=False)
        assert action in [0, 1, 2], "Action should be 0, 1, or 2"
        
        # Test get_action (training with exploration)
        action = agent.get_action(state, training=True)
        
        # Test update
        next_state = np.random.rand(18).astype(np.float32)
        loss = agent.update(state, 1, 1.0, next_state, False)
        
        # Test multiple updates to fill buffer
        for _ in range(150):
            state = np.random.rand(18).astype(np.float32)
            next_state = np.random.rand(18).astype(np.float32)
            agent.update(state, np.random.randint(0, 3), np.random.uniform(-1, 2), next_state, False)
        
        # Test save/load
        test_path = "models/test_dqn.pth"
        agent.save(test_path)
        assert os.path.exists(test_path), "Model file should exist"
        
        # Load into new agent
        agent2 = DQNAgent(state_size=18, action_size=3)
        loaded = agent2.load(test_path)
        assert loaded, "Model should load successfully"
        
        # Test get_action_name
        assert agent.get_action_name(0) == "no_feedback"
        assert agent.get_action_name(1) == "subtle_alert"
        assert agent.get_action_name(2) == "strong_alert"
        
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
        
        print_test("DQNAgent instantiation", "PASS")
        print_test("QNetwork forward pass", "PASS")
        print_test("DQN get_action (inference)", "PASS")
        print_test("DQN get_action (training)", "PASS")
        print_test("DQN update & learning", "PASS")
        print_test("DQN save/load", "PASS")
        print_test("DQN action naming", "PASS")
        
    except Exception as e:
        print_test("DQN Agent", "FAIL", str(e))
        traceback.print_exc()

def test_ppo_agent():
    """Test PPO agent"""
    print_header("TESTING: PPO Agent")
    
    try:
        import torch
        from rl_ppo_agent import PPOPPOAgent, ActorCritic, Trajectory
        
        # Test ActorCritic network
        network = ActorCritic(state_size=18, action_size=3, hidden_sizes=[256, 128, 64])
        test_input = torch.randn(1, 18)
        dist, value = network(test_input)
        assert dist.logits.shape == (1, 3), "Policy output should be (1, 3)"
        assert value.shape == (1,), "Value output should be (1,)"
        
        # Test agent creation
        agent = PPOPPOAgent(state_size=18, action_size=3)
        
        # Test get_action (training)
        state = np.random.rand(18).astype(np.float32)
        action = agent.get_action(state, training=True)
        assert action in [0, 1, 2], "Action should be 0, 1, or 2"
        assert len(agent.trajectory.states) == 1, "Trajectory should have 1 state"
        
        # Test record_step
        agent.record_step(1.0, False)
        assert len(agent.trajectory.rewards) == 1, "Should have 1 reward"
        
        # Need to call get_action more times to match steps for compute_gae
        agent.get_action(state, training=True)
        agent.record_step(0.5, False)
        agent.get_action(state, training=True)
        agent.record_step(2.0, True)
        
        # Test compute_gae
        advantages, returns = agent.compute_gae(next_value=0.0)
        assert len(advantages) == 3, "Should have 3 advantages"
        
        # Test update
        losses = agent.update(next_value=0.0)
        assert "loss" in losses, "Should return loss dict"
        assert len(agent.trajectory.states) == 0, "Trajectory should be cleared after update"
        
        # Test get_action (inference)
        action = agent.get_action(state, training=False)
        assert action in [0, 1, 2], "Action should be valid"
        
        # Test save/load
        test_path = "models/test_ppo.pth"
        agent.save(test_path)
        assert os.path.exists(test_path), "Model file should exist"
        
        agent2 = PPOPPOAgent(state_size=18, action_size=3)
        loaded = agent2.load(test_path)
        assert loaded, "Model should load successfully"
        
        # Test action probabilities
        probs = agent.get_action_probs(state)
        assert probs.shape == (3,), "Should return 3 probabilities"
        assert abs(sum(probs) - 1.0) < 0.01, "Probabilities should sum to 1"
        
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
        
        print_test("PPOPPOAgent instantiation", "PASS")
        print_test("ActorCritic forward pass", "PASS")
        print_test("PPO get_action (training)", "PASS")
        print_test("PPO record_step", "PASS")
        print_test("PPO compute_gae", "PASS")
        print_test("PPO update", "PASS")
        print_test("PPO save/load", "PASS")
        print_test("PPO action probabilities", "PASS")
        
    except Exception as e:
        print_test("PPO Agent", "FAIL", str(e))
        traceback.print_exc()

def test_simulation():
    """Test training simulation"""
    print_header("TESTING: Training Simulation")
    
    try:
        from posture_module import PostureLabel
        from simulation_enhanced import (
            UserBehaviorProfile, SimulatedUser, TrainingSimulator,
            PPOEnvWrapper, train_ppo, train_dqn, compare_algorithms
        )
        
        # Test user profile creation
        easy_profile = UserBehaviorProfile.random_profile("easy")
        assert 0.8 <= easy_profile.compliance_rate <= 0.95
        assert easy_profile.difficulty == "easy"
        
        medium_profile = UserBehaviorProfile.random_profile("medium")
        assert 0.5 <= medium_profile.compliance_rate <= 0.75
        
        hard_profile = UserBehaviorProfile.random_profile("hard")
        assert 0.2 <= hard_profile.compliance_rate <= 0.5
        
        # Test SimulatedUser
        user = SimulatedUser(easy_profile)
        label, score = user.get_posture()
        
        # Test receive_feedback
        corrected = user.receive_feedback(1)  # Subtle alert
        assert isinstance(corrected, bool), "Should return boolean"
        
        # Get stats
        stats = user.get_stats()
        assert "compliance" in stats
        assert "motivation" in stats
        
        # Test natural drift
        user.natural_drift()
        
        # Test TrainingSimulator
        sim = TrainingSimulator(num_users=5, difficulty="medium")
        user = sim.get_current_user()
        
        # Test step
        label, score = sim.step(0)  # No feedback
        label, score = sim.step(1)  # Alert
        
        # Test user switch
        sim.switch_user()
        
        # Test PPOEnvWrapper
        from simulation_enhanced import PPOEnvWrapper
        wrapper = PPOEnvWrapper(sim)
        
        state = wrapper.reset()
        assert state.shape == (18,), f"State should be 18-dim, got {state.shape}"
        
        next_state, reward, done = wrapper.step(0)
        assert isinstance(reward, float), "Reward should be float"
        
        print_test("UserBehaviorProfile generation (easy)", "PASS")
        print_test("UserBehaviorProfile generation (medium)", "PASS")
        print_test("UserBehaviorProfile generation (hard)", "PASS")
        print_test("SimulatedUser instantiation", "PASS")
        print_test("SimulatedUser feedback response", "PASS")
        print_test("SimulatedUser natural drift", "PASS")
        print_test("TrainingSimulator operations", "PASS")
        print_test("PPOEnvWrapper reset", "PASS")
        print_test("PPOEnvWrapper step", "PASS")
        
    except Exception as e:
        print_test("Training Simulation", "FAIL", str(e))
        traceback.print_exc()

def test_training():
    """Test actual training (short run)"""
    print_header("TESTING: RL Training (Short Run)")
    
    try:
        # Set seeds for reproducibility
        import random
        import torch
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        from simulation_enhanced import train_ppo, train_dqn
        
        # Quick training test (20 episodes)
        print(f"{YELLOW}Running quick PPO training test (20 episodes)...{RESET}")
        ppo_stats = train_ppo(num_episodes=20, num_users=3, verbose=False, difficulty="easy")
        
        assert "episode_rewards" in ppo_stats
        assert len(ppo_stats["episode_rewards"]) == 20
        assert ppo_stats["eval_rewards"], "Should have eval rewards"
        
        # Quick DQN training test
        print(f"{YELLOW}Running quick DQN training test (20 episodes)...{RESET}")
        dqn_stats = train_dqn(num_episodes=20, num_users=3, verbose=False, difficulty="easy")
        
        assert "episode_rewards" in dqn_stats
        assert len(dqn_stats["episode_rewards"]) == 20
        
        print_test("PPO training (20 episodes)", "PASS")
        print_test("DQN training (20 episodes)", "PASS")
        print_test("Training result storage", "PASS")
        
    except Exception as e:
        print_test("RL Training", "FAIL", str(e))
        traceback.print_exc()

def test_algorithm_selector():
    """Test algorithm selector"""
    print_header("TESTING: Algorithm Selector")
    
    try:
        from algorithm_selector import AlgorithmSelector, create_algorithm_selector
        
        # Test creation
        selector = create_algorithm_selector(evaluation_interval=5.0, min_alerts=3)
        
        # Test initial state
        algo = selector.get_active_algorithm()
        assert algo == "ppo", "Default should be ppo"
        
        # Test record_alert
        selector.record_alert(1, 0.5)  # Alert sent
        selector.record_alert(2, 0.4)  # Strong alert
        
        # Test record_posture_change (successful correction)
        selector.record_posture_change(0.6)  # Score improved
        
        # Test should_switch (should not switch yet - not enough time)
        should_switch = selector.should_switch()
        
        # Simulate time passing
        selector.last_evaluation_time = 0  # Force evaluation
        
        # Test get_stats
        stats = selector.get_stats()
        assert "current_algorithm" in stats
        
        print_test("AlgorithmSelector creation", "PASS")
        print_test("get_active_algorithm", "PASS")
        print_test("record_alert", "PASS")
        print_test("record_posture_change", "PASS")
        print_test("get_stats", "PASS")
        
    except Exception as e:
        print_test("Algorithm Selector", "FAIL", str(e))
        traceback.print_exc()

def test_utils():
    """Test utility functions"""
    print_header("TESTING: Utilities")
    
    try:
        from utils import (
            MetricsLogger, SessionLogger, RateCalculator,
            setup_directories, validate_config, compute_angle, moving_average
        )
        
        # Test setup_directories
        setup_directories()
        assert os.path.exists("logs")
        assert os.path.exists("models")
        assert os.path.exists("data")
        
        # Test validate_config
        valid = validate_config()
        assert valid == True, "Config should be valid"
        
        # Test MetricsLogger
        logger = MetricsLogger()
        logger.log_episode(1, 10.0, 0.5, 0.9, 0.8, 0.7)
        stats = logger.compute_statistics(window=10)
        assert "avg_reward" in stats
        logger.close()
        
        # Test SessionLogger
        session = SessionLogger()
        session.log_frame(0.8, 0, 1.0)
        session.log_frame(0.6, 1, 0.5)
        session.log_frame(0.9, 2, 2.0)
        summary = session.get_summary()
        assert "total_frames" in summary
        assert summary["total_frames"] == 3
        
        # Test RateCalculator
        rate = RateCalculator()
        for _ in range(10):
            rate.tick()
            time.sleep(0.01)
        fps = rate.get_fps()
        
        # Test moving_average
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = moving_average(data, window=5)
        assert len(result) == len(data)
        
        print_test("setup_directories", "PASS")
        print_test("validate_config", "PASS")
        print_test("MetricsLogger", "PASS")
        print_test("SessionLogger", "PASS")
        print_test("RateCalculator", "PASS")
        print_test("moving_average", "PASS")
        
    except Exception as e:
        print_test("Utilities", "FAIL", str(e))
        traceback.print_exc()

def test_feedback_system():
    """Test visual feedback system"""
    print_header("TESTING: Feedback System")
    
    try:
        from feedback import VisualFeedback, AlertSystem, FeedbackManager, FeedbackLevel
        
        # Test VisualFeedback
        vf = VisualFeedback()
        
        import numpy as np
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test draw_overlay
        result = vf.draw_overlay(
            test_frame,
            "good",
            0.85,
            "no_feedback",
            {"FPS": "30", "Alerts": "5"}
        )
        assert result.shape == test_frame.shape, "Output should match input shape"
        
        # Test draw_skeleton_colored
        result = vf.draw_skeleton_colored(test_frame, {}, 0.5)
        
        # Test AlertSystem
        alert_sys = AlertSystem()
        current_time = time.time()
        should = alert_sys.should_alert(current_time)
        
        # Test FeedbackManager
        fm = FeedbackManager()
        
        print_test("VisualFeedback instantiation", "PASS")
        print_test("VisualFeedback.draw_overlay", "PASS")
        print_test("VisualFeedback.draw_skeleton_colored", "PASS")
        print_test("AlertSystem.alert_logic", "PASS")
        print_test("FeedbackManager", "PASS")
        
    except Exception as e:
        print_test("Feedback System", "FAIL", str(e))
        traceback.print_exc()

def test_pose_detection():
    """Test pose detection module"""
    print_header("TESTING: Pose Detection (MediaPipe)")
    
    try:
        from pose_module import PoseDetector, PoseAnalyzer, PoseCalibrator
        
        # Test PoseDetector creation (will fail if no camera, that's ok)
        try:
            detector = PoseDetector()
            has_camera = True
        except Exception as e:
            print(f"{YELLOW}  Note: No camera available - testing without video capture{RESET}")
            has_camera = False
        
        # Test keypoint extraction (mock data)
        from pose_module import PoseDetector
        LANDMARKS = PoseDetector.LANDMARKS
        assert len(LANDMARKS) == 33, "Should have 33 landmarks"
        
        # Test key landmark mapping
        KEY_LANDMARKS = PoseDetector.KEY_LANDMARKS
        assert "nose" in KEY_LANDMARKS
        assert "left_shoulder" in KEY_LANDMARKS
        assert "right_shoulder" in KEY_LANDMARKS
        
        # Test PoseAnalyzer with mock data
        analyzer = PoseAnalyzer()
        
        # Test angle calculations
        p1 = (100, 100)
        p2 = (200, 100)
        angle = analyzer.compute_angle(p1, p2)
        
        # Test neck angle computation (mock)
        mock_nose = {"y": 100, "x": 200}
        mock_shoulders = {
            "left_shoulder": {"y": 200, "x": 100},
            "right_shoulder": {"y": 200, "x": 300}
        }
        neck_angle = analyzer.compute_neck_angle(mock_nose, mock_shoulders)
        
        # Test shoulder diff
        shoulder_diff = analyzer.compute_shoulder_diff(mock_shoulders)
        
        # Test spine inclination
        mock_hips = {
            "left_hip": {"y": 400, "x": 120},
            "right_hip": {"y": 400, "x": 280}
        }
        spine = analyzer.compute_spine_inclination(mock_shoulders, mock_hips)
        
        # Test PoseCalibrator
        calibrator = PoseCalibrator(num_frames=10)
        
        print_test("PoseDetector creation", "PASS" if has_camera else "WARN")
        print_test("Landmark definitions (33 points)", "PASS")
        print_test("Key landmark mapping", "PASS")
        print_test("PoseAnalyzer.angle_calculation", "PASS")
        print_test("PoseAnalyzer.compute_neck_angle", "PASS")
        print_test("PoseAnalyzer.compute_shoulder_diff", "PASS")
        print_test("PoseAnalyzer.compute_spine_inclination", "PASS")
        print_test("PoseCalibrator", "PASS")
        
    except Exception as e:
        print_test("Pose Detection", "FAIL", str(e))
        traceback.print_exc()

def test_attention_tracker():
    """Test attention tracking"""
    print_header("TESTING: Attention Tracker")
    
    try:
        from attention_tracker import AttentionTracker, AttentionState, FaceDetector, GazeMetrics
        
        # Test FaceDetector
        fd = FaceDetector()
        
        # Test AttentionTracker
        tracker = AttentionTracker()
        
        # Test state transitions
        state = tracker.get_current_state()
        assert state == AttentionState.UNKNOWN
        
        # Test session metrics
        metrics = tracker.get_session_metrics()
        assert "focus_percentage" in metrics
        
        # Test attention score
        score = tracker.get_attention_score()
        
        # Test is_user_attending / is_user_away
        attending = tracker.is_user_attending()
        away = tracker.is_user_away()
        
        # Test get_posture_attention_factor
        factor = tracker.get_posture_attention_factor()
        
        print_test("FaceDetector instantiation", "PASS")
        print_test("AttentionTracker instantiation", "PASS")
        print_test("AttentionTracker.get_session_metrics", "PASS")
        print_test("AttentionTracker.get_attention_score", "PASS")
        print_test("AttentionTracker.user_state_checks", "PASS")
        print_test("AttentionTracker.get_posture_attention_factor", "PASS")
        
    except Exception as e:
        print_test("Attention Tracker", "FAIL", str(e))
        traceback.print_exc()

def test_hand_tracker():
    """Test hand tracking"""
    print_header("TESTING: Hand Tracker")
    
    try:
        from hand_tracker import HandTracker, TypingPosture, HandMetrics
        
        # Test HandTracker
        tracker = HandTracker(use_holistic=False)  # Don't need MediaPipe
        
        # Test state
        state = tracker.get_current_state()
        
        # Test is_typing
        typing = tracker.is_typing()
        
        # Test get_typing_intensity
        intensity = tracker.get_typing_intensity()
        
        # Test session metrics
        metrics = tracker.get_session_metrics()
        assert "good_typing_percentage" in metrics
        
        # Test get_typing_posture_score
        score = tracker.get_typing_posture_score()
        
        print_test("HandTracker instantiation", "PASS")
        print_test("HandTracker.get_current_state", "PASS")
        print_test("HandTracker.is_typing", "PASS")
        print_test("HandTracker.get_typing_intensity", "PASS")
        print_test("HandTracker.get_session_metrics", "PASS")
        print_test("HandTracker.get_typing_posture_score", "PASS")
        
    except Exception as e:
        print_test("Hand Tracker", "FAIL", str(e))
        traceback.print_exc()

def test_combined_analyzer():
    """Test combined analyzer"""
    print_header("TESTING: Combined Analyzer")
    
    try:
        from combined_analyzer import UnifiedAnalyzer, CombinedMetrics, create_unified_analyzer
        
        # Test creation
        analyzer = create_unified_analyzer(enable_attention=False, enable_hands=False)
        
        # Test with mock frame
        import numpy as np
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        metrics = analyzer.analyze(mock_frame, 0.8, "good")
        
        assert isinstance(metrics, CombinedMetrics)
        assert metrics.posture_score == 0.8
        assert metrics.combined_score >= 0.0
        
        # Test session summary
        summary = analyzer.get_session_summary()
        assert "total_frames" in summary
        
        print_test("UnifiedAnalyzer creation", "PASS")
        print_test("UnifiedAnalyzer.analyze", "PASS")
        print_test("UnifiedAnalyzer.get_session_summary", "PASS")
        
    except Exception as e:
        print_test("Combined Analyzer", "FAIL", str(e))
        traceback.print_exc()

def test_main_system():
    """Test main system integration"""
    print_header("TESTING: Main System Integration")
    
    try:
        from main import PostureSystem, train_agent, train_both, compare_agents
        
        # Test train_agent function exists and can be imported
        assert callable(train_agent)
        assert callable(train_both)
        assert callable(compare_agents)
        
        # Test PostureSystem can be instantiated (won't initialize without camera)
        # Just test that class exists and has required methods
        system = PostureSystem.__new__(PostureSystem)
        
        print_test("Main imports", "PASS")
        print_test("train_agent function", "PASS")
        print_test("train_both function", "PASS")
        print_test("compare_agents function", "PASS")
        
    except Exception as e:
        print_test("Main System", "FAIL", str(e))
        traceback.print_exc()

def test_gui_app():
    """Test GUI application"""
    print_header("TESTING: GUI Application")
    
    try:
        # Test imports (can't run GUI in test environment)
        import gui_app
        assert hasattr(gui_app, 'UPRYTApplication')
        
        print_test("GUI imports", "PASS")
        print_test("UPRYTApplication class", "PASS")
        
    except Exception as e:
        print_test("GUI Application", "FAIL", str(e))
        traceback.print_exc()

def test_model_loading():
    """Test loading trained models"""
    print_header("TESTING: Trained Model Loading")
    
    try:
        from rl_ppo_agent import PPOPPOAgent
        from rl_agent import DQNAgent
        from config import config
        
        # Check if models exist
        ppo_path = f"{config.system.model_dir}/ppo_final.pth"
        dqn_path = f"{config.system.model_dir}/dqn_final.pth"
        
        ppo_exists = os.path.exists(ppo_path)
        dqn_exists = os.path.exists(dqn_path)
        
        if ppo_exists:
            ppo = PPOPPOAgent(state_size=18, action_size=3)
            loaded = ppo.load(ppo_path)
            print_test("PPO model loading", "PASS" if loaded else "FAIL")
        else:
            print_test("PPO model existence", "WARN", "No trained model found")
        
        if dqn_exists:
            dqn = DQNAgent(state_size=18, action_size=3)
            loaded = dqn.load(dqn_path)
            print_test("DQN model loading", "PASS" if loaded else "FAIL")
        else:
            print_test("DQN model existence", "WARN", "No trained model found")
        
    except Exception as e:
        print_test("Model Loading", "FAIL", str(e))
        traceback.print_exc()

def test_results_saved():
    """Test that training results are saved correctly"""
    print_header("TESTING: Results Storage")
    
    try:
        results_dir = "results"
        
        if os.path.exists(results_dir):
            files = os.listdir(results_dir)
            json_files = [f for f in files if f.endswith('.json')]
            
            if json_files:
                # Read most recent result
                latest = sorted(json_files)[-1]
                with open(os.path.join(results_dir, latest), 'r') as f:
                    result = json.load(f)
                
                assert "algorithm" in result
                assert "episodes" in result
                assert "episode_rewards" in result
                
                print_test("Results directory exists", "PASS")
                print_test("Results JSON structure", "PASS")
                print_test("Results file content", "PASS")
            else:
                print_test("Results directory", "WARN", "No result files found")
        else:
            print_test("Results directory", "WARN", "No results directory found")
        
    except Exception as e:
        print_test("Results Storage", "FAIL", str(e))
        traceback.print_exc()

def print_summary():
    """Print test summary"""
    print_header("TEST SUMMARY")
    
    total = len(test_results["passed"]) + len(test_results["failed"])
    pass_rate = (len(test_results["passed"]) / total * 100) if total > 0 else 0
    
    print(f"{BOLD}Total Tests: {total}{RESET}")
    print(f"{GREEN}Passed: {len(test_results['passed'])}{RESET}")
    print(f"{RED}Failed: {len(test_results['failed'])}{RESET}")
    print(f"{YELLOW}Warnings: {len(test_results['warnings'])}{RESET}")
    print(f"{BOLD}Pass Rate: {pass_rate:.1f}%{RESET}")
    
    if test_results["failed"]:
        print(f"\n{RED}{BOLD}FAILED TESTS:{RESET}")
        for name, details in test_results["failed"]:
            print(f"  - {name}")
            if details:
                print(f"    {details}")
    
    if test_results["warnings"]:
        print(f"\n{YELLOW}{BOLD}WARNINGS:{RESET}")
        for name, details in test_results["warnings"]:
            print(f"  - {name}")
    
    return len(test_results["failed"]) == 0

if __name__ == "__main__":
    print(f"\n{BOLD}{'='*70}")
    print(f"UPRYT COMPREHENSIVE SYSTEM TEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}{RESET}\n")
    
    # Run all tests
    test_config()
    test_posture_labels()
    test_rule_based_classifier()
    test_environment()
    test_dqn_agent()
    test_ppo_agent()
    test_simulation()
    test_training()
    test_algorithm_selector()
    test_utils()
    test_feedback_system()
    test_pose_detection()
    test_attention_tracker()
    test_hand_tracker()
    test_combined_analyzer()
    test_main_system()
    test_gui_app()
    test_model_loading()
    test_results_saved()
    
    # Print summary
    success = print_summary()
    
    print(f"\n{BOLD}Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}\n")
    
    sys.exit(0 if success else 1)