import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from posture_module import PostureLabel, encode_label
from environment import Action, PostureState, PostureEnvironment
from config import config
import random
import os
import json
from datetime import datetime


@dataclass
class UserBehaviorProfile:
    compliance_rate: float = 0.7
    fatigue_threshold: int = 3
    stubbornness: float = 0.2
    attention_span: float = 0.8
    learning_rate: float = 0.05
    reset_probability: float = 0.1
    posture_preference: str = "neutral"
    difficulty: str = "medium"
    
    @classmethod
    def random_profile(cls, difficulty: str = "medium") -> "UserBehaviorProfile":
        if difficulty == "easy":
            return cls(
                compliance_rate=random.uniform(0.8, 0.95),
                fatigue_threshold=random.randint(5, 8),
                stubbornness=random.uniform(0.05, 0.15),
                attention_span=random.uniform(0.85, 1.0),
                learning_rate=random.uniform(0.1, 0.2),
                reset_probability=0.03,
                posture_preference=random.choice(["good", "neutral"]),
                difficulty="easy",
            )
        elif difficulty == "medium":
            return cls(
                compliance_rate=random.uniform(0.5, 0.75),
                fatigue_threshold=random.randint(2, 4),
                stubbornness=random.uniform(0.15, 0.35),
                attention_span=random.uniform(0.6, 0.85),
                learning_rate=random.uniform(0.05, 0.1),
                reset_probability=0.1,
                posture_preference=random.choice(["good", "neutral", "slouching"]),
                difficulty="medium",
            )
        else:  # hard
            return cls(
                compliance_rate=random.uniform(0.2, 0.5),
                fatigue_threshold=random.randint(1, 2),
                stubbornness=random.uniform(0.35, 0.6),
                attention_span=random.uniform(0.3, 0.6),
                learning_rate=random.uniform(0.01, 0.04),
                reset_probability=0.25,
                posture_preference=random.choice(["slouching", "leaning", "forward_head"]),
                difficulty="hard",
            )

    def to_dict(self) -> dict:
        return {
            "compliance_rate": self.compliance_rate,
            "fatigue_threshold": self.fatigue_threshold,
            "stubbornness": self.stubbornness,
            "attention_span": self.attention_span,
            "learning_rate": self.learning_rate,
            "reset_probability": self.reset_probability,
            "posture_preference": self.posture_preference,
            "difficulty": self.difficulty,
        }


class SimulatedUser:
    def __init__(self, profile: UserBehaviorProfile = None, user_id: str = None):
        self.profile = profile or UserBehaviorProfile()
        self.user_id = user_id or f"user_{random.randint(1000, 9999)}"
        self.current_posture: PostureLabel = PostureLabel.GOOD
        self.current_score: float = 0.8
        self.consecutive_alerts: int = 0
        self.total_alerts_received: int = 0
        self.total_corrections: int = 0
        self.correction_streak: int = 0
        self.max_correction_streak: int = 0
        self.behavior_state = {
            "fatigue_level": 0.0,
            "motivation": random.uniform(0.7, 1.0),
            "frustration": 0.0,
        }
        self.episode_alerts = 0
        self.episode_corrections = 0

    def reset_episode(self):
        self.consecutive_alerts = 0
        self.episode_alerts = 0
        self.episode_corrections = 0

    def get_posture(self) -> Tuple[PostureLabel, float]:
        return self.current_posture, self.current_score

    def receive_feedback(self, action: int) -> bool:
        if action == Action.NO_FEEDBACK.value:
            return False
        
        self.total_alerts_received += 1
        self.consecutive_alerts += 1
        self.episode_alerts += 1
        
        correction_made = self._simulate_response(action)
        
        if correction_made:
            self.total_corrections += 1
            self.episode_corrections += 1
            self.correction_streak += 1
            self.max_correction_streak = max(self.max_correction_streak, self.correction_streak)
            self.behavior_state["motivation"] = min(1.0, self.behavior_state["motivation"] + 0.02)
            self.behavior_state["frustration"] = max(0.0, self.behavior_state["frustration"] - 0.05)
            self._improve_posture()
        else:
            self.correction_streak = 0
            self.behavior_state["frustration"] = min(1.0, self.behavior_state["frustration"] + 0.08)
            if self.behavior_state["frustration"] > 0.6:
                self.behavior_state["motivation"] = max(0.0, self.behavior_state["motivation"] - 0.05)
            self._worsen_posture()
        
        return correction_made

    def _simulate_response(self, action: int) -> bool:
        base = self.profile.compliance_rate
        
        # Motivation increases compliance
        base *= (0.5 + 0.5 * self.behavior_state["motivation"])
        
        # Frustration decreases compliance  
        base *= (1.0 - 0.4 * self.behavior_state["frustration"])
        
        # Fatigue from too many consecutive alerts
        if self.consecutive_alerts > self.profile.fatigue_threshold:
            fatigue_penalty = (self.consecutive_alerts - self.profile.fatigue_threshold) * 0.15
            base *= max(0.1, 1.0 - fatigue_penalty)
        
        # Stronger alerts work better
        if action == Action.STRONG_ALERT.value:
            base += 0.15
        elif action == Action.SUBTLE_ALERT.value:
            base += 0.05
        
        # First alert of a session gets attention bonus
        if self.consecutive_alerts == 1:
            base *= (0.7 + 0.3 * self.profile.attention_span)
        
        # User's posture preference affects compliance
        if self.profile.posture_preference == "slouching":
            base *= 0.65
        elif self.profile.posture_preference == "leaning":
            base *= 0.7
        elif self.profile.posture_preference == "forward_head":
            base *= 0.7
        
        base = max(0.05, min(0.95, base))
        return random.random() < base

    def _improve_posture(self):
        self.consecutive_alerts = 0
        improvement = 0.2 + self.profile.learning_rate * 0.15
        
        if self.current_posture == PostureLabel.GOOD:
            self.current_score = min(0.95, self.current_score + 0.03)
        else:
            self.current_posture = PostureLabel.GOOD
            self.current_score = min(0.9, self.current_score + improvement)

    def _worsen_posture(self):
        decay = 0.1 + self.profile.stubbornness * 0.12
        decay *= (1.0 + 0.3 * self.behavior_state["frustration"])
        self.current_score = max(0.2, self.current_score - decay)
        
        # Occasionally change posture type
        if self.current_score < 0.6 and random.random() < 0.4:
            bad_postures = [PostureLabel.SLOUCHING, PostureLabel.FORWARD_HEAD, PostureLabel.LEANING]
            self.current_posture = random.choice(bad_postures)

    def natural_drift(self):
        self.behavior_state["fatigue_level"] = min(1.0, self.behavior_state["fatigue_level"] + 0.008)
        
        # Natural score decay
        if random.random() < 0.2 * (1 + self.behavior_state["fatigue_level"]):
            self.current_score = max(0.3, self.current_score - 0.02)
            
            # Drift to bad posture if score low
            if self.current_score < 0.55 and self.current_posture == PostureLabel.GOOD:
                if random.random() < 0.4:
                    self.current_posture = random.choice([
                        PostureLabel.SLOUCHING, 
                        PostureLabel.FORWARD_HEAD, 
                        PostureLabel.LEANING
                    ])
        
        # Occasional motivation boost
        if random.random() < 0.02:
            self.behavior_state["motivation"] = min(1.0, self.behavior_state["motivation"] + 0.08)
            self.behavior_state["frustration"] = max(0.0, self.behavior_state["frustration"] - 0.03)

    def get_stats(self) -> dict:
        return {
            "compliance": self.total_corrections / max(1, self.total_alerts_received),
            "streak": self.correction_streak,
            "max_streak": self.max_correction_streak,
            "fatigue": self.behavior_state["fatigue_level"],
            "frustration": self.behavior_state["frustration"],
            "motivation": self.behavior_state["motivation"],
        }

    def reset_user(self):
        self.current_posture = PostureLabel.GOOD
        self.current_score = 0.8
        self.consecutive_alerts = 0
        self.behavior_state["fatigue_level"] = max(0, self.behavior_state["fatigue_level"] - 0.15)
        self.behavior_state["frustration"] = max(0, self.behavior_state["frustration"] - 0.1)


class TrainingSimulator:
    def __init__(self, num_users: int = 8, difficulty: str = "medium"):
        self.num_users = num_users
        self.difficulty = difficulty
        
        # Create diverse user pool
        self.users = [SimulatedUser(UserBehaviorProfile.random_profile(difficulty)) 
                      for _ in range(num_users)]
        self.current_user_idx = 0
        
    def switch_user(self):
        self.current_user_idx = (self.current_user_idx + 1) % self.num_users
        
        # Randomly regenerate some users for diversity
        if random.random() < 0.25:
            new_profile = UserBehaviorProfile.random_profile(self.difficulty)
            self.users[self.current_user_idx] = SimulatedUser(new_profile)
        
        return self.get_current_user()
    
    def get_current_user(self) -> SimulatedUser:
        return self.users[self.current_user_idx]
    
    def reset_current_user(self):
        self.get_current_user().reset_user()
        self.get_current_user().reset_episode()
    
    def step(self, action: int) -> Tuple[PostureLabel, float]:
        user = self.get_current_user()
        
        if action != Action.NO_FEEDBACK.value:
            user.receive_feedback(action)
        
        user.natural_drift()
        
        return user.get_posture()


class PPOEnvWrapper:
    """Environment wrapper that provides proper rewards for RL training"""
    
    def __init__(self, simulator: TrainingSimulator):
        self.simulator = simulator
        self.current_state = None
        
    def reset(self) -> np.ndarray:
        if random.random() < 0.3:
            self.simulator.switch_user()
        self.simulator.reset_current_user()
        
        label, score = self.simulator.get_current_user().get_posture()
        
        # Start with random bad posture 40% of time
        if random.random() < 0.4:
            self.simulator.get_current_user().current_posture = random.choice([
                PostureLabel.SLOUCHING, PostureLabel.FORWARD_HEAD, PostureLabel.LEANING
            ])
            self.simulator.get_current_user().current_score = random.uniform(0.3, 0.55)
        
        self.current_state = self._create_state(label, score)
        return self.current_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        user = self.simulator.get_current_user()
        
        # Get posture BEFORE action (for reward calculation)
        old_label, old_score = user.get_posture()
        was_bad = (old_label != PostureLabel.GOOD)
        
        # Execute action
        label, score = self.simulator.step(action)
        
        # Compute reward
        reward = self._compute_reward(action, was_bad, label, score)
        
        # Create new state
        self.current_state = self._create_state(label, score)
        
        # Done when user reaches high score or max steps
        done = score >= 0.9
        
        return self.current_state, reward, done
    
    def _create_state(self, label: PostureLabel, score: float) -> np.ndarray:
        user = self.simulator.get_current_user()
        
        # Build 18-dim state vector
        state = np.array([
            # Basic features (6)
            encode_label(label),
            score,
            user.consecutive_alerts / 5.0,
            user.behavior_state["fatigue_level"],
            user.behavior_state["motivation"],
            user.behavior_state["frustration"],
            # Derived features (6)
            1.0 if label == PostureLabel.GOOD else 0.0,
            1.0 - score,  # inverse score = badness
            user.episode_alerts / 20.0,
            user.episode_corrections / max(1, user.episode_alerts) if user.episode_alerts > 0 else 0.0,
            user.correction_streak / 10.0,
            user.max_correction_streak / 20.0,
            # User profile features (6)
            user.profile.compliance_rate,
            user.profile.stubbornness,
            user.profile.attention_span,
            user.profile.learning_rate,
            1.0 if user.profile.difficulty == "easy" else 0.0,
            1.0 if user.profile.difficulty == "hard" else 0.0,
        ], dtype=np.float32)
        
        return state
    
    def _compute_reward(self, action: int, was_bad: bool, label: PostureLabel, 
                        score: float) -> float:
        user = self.simulator.get_current_user()
        is_good = (label == PostureLabel.GOOD)
        
        if action == Action.NO_FEEDBACK.value:
            if is_good:
                return 0.05 * score  # Small reward for maintaining
            else:
                return -0.1 * (1.0 - score)  # Reduced penalty
        
        # Alert given
        if was_bad and is_good:
            # Successful correction!
            base = 1.5  # reduced from 2.0
            if action == Action.STRONG_ALERT.value:
                base = 1.8  # reduced from 2.5
            # TIME BONUS: faster correction = more reward
            if user.consecutive_alerts <= 2:
                base += 0.5  # bonus for quick correction
            return base
        elif not was_bad and is_good:
            # Good posture maintained with alert
            return 0.2  # reduced from 0.3
        else:
            # Alert ignored
            return -0.3  # reduced from -0.8


class UnifiedTrainer:
    def __init__(self, agent, simulator: TrainingSimulator = None,
                 num_episodes: int = 500, algorithm: str = "ppo"):
        self.agent = agent
        self.simulator = simulator or TrainingSimulator(num_users=8, difficulty="medium")
        self.env_wrapper = PPOEnvWrapper(self.simulator)
        self.num_episodes = num_episodes
        self.algorithm = algorithm.lower()
        
        self.episode_rewards = []
        self.eval_rewards = []
        self.best_reward = float("-inf")

    def train(self, eval_interval: int = 50, verbose: bool = True) -> Dict[str, Any]:
        stats = {
            "algorithm": self.algorithm,
            "episode_rewards": [],
            "eval_rewards": [],
            "train_rewards": [],
        }
        
        for episode in range(self.num_episodes):
            # Run episode
            episode_reward = self._run_episode()
            self.episode_rewards.append(episode_reward)
            stats["episode_rewards"].append(episode_reward)
            
            # Evaluate periodically
            if episode % eval_interval == 0 or episode == self.num_episodes - 1:
                eval_reward = self._evaluate(num_eval=5)
                self.eval_rewards.append(eval_reward)
                stats["eval_rewards"].append(eval_reward)
                
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self._save_model(f"models/{self.algorithm}_best.pth")
                
                if verbose:
                    recent = np.mean(self.episode_rewards[-eval_interval:]) if len(self.episode_rewards) >= eval_interval else 0
                    print(f"[{self.algorithm.upper()}] Ep {episode}: Train={episode_reward:.2f}, "
                          f"Eval={eval_reward:.2f}, RecentAvg={recent:.2f}")
            
            # Save checkpoint
            if episode > 0 and episode % 200 == 0:
                self._save_model(f"models/{self.algorithm}_ep{episode}.pth")
        
        self._save_model(f"models/{self.algorithm}_final.pth")
        return stats

    def _run_episode(self) -> float:
        state = self.env_wrapper.reset()
        episode_reward = 0.0
        steps = 0
        max_steps = 150
        
        while steps < max_steps:
            action = self.agent.get_action(state, training=True)
            next_state, reward, done = self.env_wrapper.step(action)
            
            if self.algorithm == "ppo":
                self.agent.record_step(reward, done)
            else:
                self.agent.update(state, action, reward, next_state, done)
                self.agent.decay_epsilon()
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        # PPO update at end of episode
        if self.algorithm == "ppo":
            self.agent.update(0.0)
        
        return episode_reward

    def _evaluate(self, num_eval: int = 5) -> float:
        eval_rewards = []
        
        for _ in range(num_eval):
            state = self.env_wrapper.reset()
            episode_reward = 0.0
            steps = 0
            max_steps = 150
            
            while steps < max_steps:
                action = self.agent.get_action(state, training=False)
                next_state, reward, done = self.env_wrapper.step(action)
                
                episode_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards)

    def _save_model(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.agent.save(path)


def _save_training_results(algo: str, stats: Dict[str, Any], num_episodes: int, 
                          num_users: int, difficulty: str):
    """Save training results to JSON file with timestamp"""
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        "algorithm": algo,
        "episodes": num_episodes,
        "users": num_users,
        "difficulty": difficulty,
        "timestamp": datetime.now().isoformat(),
        "episode_rewards": stats.get("episode_rewards", []),
        "eval_rewards": stats.get("eval_rewards", []),
        "final_eval_reward": stats.get("eval_rewards", [0])[-1] if stats.get("eval_rewards") else 0,
        "best_eval_reward": max(stats.get("eval_rewards", [0])) if stats.get("eval_rewards") else 0,
    }
    
    filename = f"results/{algo}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved training results to: {filename}")


def train_ppo(num_episodes: int = 2000, num_users: int = 8, verbose: bool = True,
              difficulty: str = "medium") -> Dict[str, Any]:
    from rl_ppo_agent import PPOPPOAgent
    agent = PPOPPOAgent(state_size=18, action_size=3)
    simulator = TrainingSimulator(num_users=num_users, difficulty=difficulty)
    trainer = UnifiedTrainer(agent, simulator, num_episodes=num_episodes, algorithm="ppo")
    stats = trainer.train(verbose=verbose)
    _save_training_results("ppo", stats, num_episodes, num_users, difficulty)
    return stats


def train_dqn(num_episodes: int = 2000, num_users: int = 8, verbose: bool = True,
              difficulty: str = "medium") -> Dict[str, Any]:
    from rl_agent import DQNAgent
    agent = DQNAgent(state_size=18, action_size=3)
    simulator = TrainingSimulator(num_users=num_users, difficulty=difficulty)
    trainer = UnifiedTrainer(agent, simulator, num_episodes=num_episodes, algorithm="dqn")
    stats = trainer.train(verbose=verbose)
    _save_training_results("dqn", stats, num_episodes, num_users, difficulty)
    return stats


def benchmark(agent, num_episodes: int = 100, difficulty: str = "medium") -> Dict[str, Any]:
    """Run benchmark to evaluate agent performance"""
    simulator = TrainingSimulator(num_users=8, difficulty=difficulty)
    env_wrapper = PPOEnvWrapper(simulator)
    
    total_alerts = 0
    total_corrections = 0
    episode_rewards = []
    
    for _ in range(num_episodes):
        state = env_wrapper.reset()
        episode_reward = 0.0
        steps = 0
        max_steps = 150
        
        while steps < max_steps:
            action = agent.get_action(state, training=False)
            
            # Count alerts
            if action != Action.NO_FEEDBACK.value:
                total_alerts += 1
            
            # Check if this was a successful correction
            old_user = simulator.get_current_user()
            old_label, _ = old_user.get_posture()
            was_bad = (old_label != PostureLabel.GOOD)
            
            next_state, reward, done = env_wrapper.step(action)
            
            # Check if correction was successful
            new_label, _ = simulator.get_current_user().get_posture()
            is_good = (new_label == PostureLabel.GOOD)
            
            if was_bad and is_good and action != Action.NO_FEEDBACK.value:
                total_corrections += 1
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
    
    return {
        "total_alerts": total_alerts,
        "total_corrections": total_corrections,
        "correction_rate": total_corrections / max(1, total_alerts),
        "avg_episode_reward": np.mean(episode_rewards),
        "avg_steps": np.mean([min(150, len(episode_rewards))]),
    }


def compare_algorithms(ppo_path: str = None, dqn_path: str = None, 
                       num_episodes: int = 50, difficulty: str = "medium") -> Dict[str, Any]:
    from rl_ppo_agent import PPOPPOAgent
    from rl_agent import DQNAgent
    
    results = {}
    
    if ppo_path and os.path.exists(ppo_path):
        ppo_agent = PPOPPOAgent(state_size=18, action_size=3)
        ppo_agent.load(ppo_path)
        results["PPO"] = benchmark(ppo_agent, num_episodes, difficulty)
    
    if dqn_path and os.path.exists(dqn_path):
        dqn_agent = DQNAgent(state_size=18, action_size=3)
        dqn_agent.load(dqn_path)
        results["DQN"] = benchmark(dqn_agent, num_episodes, difficulty)
    
    return results


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("Training PPO Agent")
    print("="*60)
    ppo_stats = train_ppo(num_episodes=200, num_users=8, verbose=True, difficulty="medium")
    print(f"PPO Final Eval: {ppo_stats['eval_rewards'][-1] if ppo_stats['eval_rewards'] else 0:.2f}")
    
    print("\n" + "="*60)
    print("Training DQN Agent")
    print("="*60)
    dqn_stats = train_dqn(num_episodes=200, num_users=8, verbose=True, difficulty="medium")
    print(f"DQN Final Eval: {dqn_stats['eval_rewards'][-1] if dqn_stats['eval_rewards'] else 0:.2f}")
    
    print("\n" + "="*60)
    print("Benchmark Comparison")
    print("="*60)
    results = compare_algorithms(
        ppo_path="models/ppo_best.pth",
        dqn_path="models/dqn_best.pth",
        num_episodes=50,
        difficulty="medium"
    )
    
    for algo, metrics in results.items():
        print(f"\n{algo}:")
        print(f"  Correction Rate: {metrics['correction_rate']:.2%}")
        print(f"  Avg Episode Reward: {metrics['avg_episode_reward']:.2f}")
        print(f"  Total Alerts: {metrics['total_alerts']}")
        print(f"  Successful Corrections: {metrics['total_corrections']}")