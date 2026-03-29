import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from posture_module import PostureLabel, encode_label
from environment import Action, PostureState, PostureEnvironment
from config import config
import random
import os
import json


@dataclass
class UserBehaviorProfile:
    compliance_rate: float = 0.7
    fatigue_threshold: int = 3
    stubbornness: float = 0.2
    attention_span: float = 0.8
    learning_rate: float = 0.05
    reset_probability: float = 0.1
    posture_preference: str = "neutral"
    
    @classmethod
    def random_profile(cls, difficulty: str = "medium") -> "UserBehaviorProfile":
        if difficulty == "easy":
            return cls(
                compliance_rate=random.uniform(0.8, 0.95),
                fatigue_threshold=random.randint(4, 6),
                stubbornness=random.uniform(0.05, 0.15),
                attention_span=random.uniform(0.85, 1.0),
                learning_rate=random.uniform(0.08, 0.15),
                reset_probability=0.05,
                posture_preference=random.choice(["good", "neutral"]),
            )
        elif difficulty == "medium":
            return cls(
                compliance_rate=random.uniform(0.5, 0.8),
                fatigue_threshold=random.randint(2, 5),
                stubbornness=random.uniform(0.1, 0.3),
                attention_span=random.uniform(0.6, 0.9),
                learning_rate=random.uniform(0.03, 0.08),
                reset_probability=0.1,
                posture_preference=random.choice(["good", "neutral", "slouching"]),
            )
        else:
            return cls(
                compliance_rate=random.uniform(0.3, 0.6),
                fatigue_threshold=random.randint(1, 3),
                stubbornness=random.uniform(0.25, 0.5),
                attention_span=random.uniform(0.4, 0.7),
                learning_rate=random.uniform(0.01, 0.04),
                reset_probability=0.2,
                posture_preference=random.choice(["slouching", "leaning", "neutral"]),
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
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserBehaviorProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SimulatedUser:
    def __init__(self, profile: UserBehaviorProfile = None, user_id: str = None):
        self.profile = profile or UserBehaviorProfile()
        self.user_id = user_id or f"user_{random.randint(1000, 9999)}"
        self.current_posture: PostureLabel = PostureLabel.GOOD
        self.current_score: float = 0.8
        self.consecutive_alerts: int = 0
        self.alert_history: list = []
        self.total_alerts_received: int = 0
        self.total_corrections: int = 0
        self.successful_alerts: int = 0
        self.ignored_alerts: int = 0
        self.session_history: List[dict] = []
        self.behavior_state = {
            "fatigue_level": 0.0,
            "motivation": random.uniform(0.5, 1.0),
            "frustration": 0.0,
            "streak": 0,
            "best_streak": 0,
        }

    def get_posture(self) -> Tuple[PostureLabel, float]:
        return self.current_posture, self.current_score

    def receive_feedback(self, action: int) -> bool:
        if action == Action.NO_FEEDBACK.value:
            return False
        self.total_alerts_received += 1
        self.consecutive_alerts += 1
        self.alert_history.append({
            "action": action,
            "score_before": self.current_score,
            "timestamp": len(self.alert_history),
        })
        correction_made = self._simulate_user_response(action)
        
        if correction_made:
            self.total_corrections += 1
            self.successful_alerts += 1
            self.behavior_state["streak"] += 1
            self.behavior_state["best_streak"] = max(self.behavior_state["best_streak"], 
                                                      self.behavior_state["streak"])
            self.behavior_state["motivation"] = min(1.0, self.behavior_state["motivation"] + 0.05)
            self.behavior_state["frustration"] = max(0.0, self.behavior_state["frustration"] - 0.1)
            self._improve_posture()
            return True
        else:
            self.ignored_alerts += 1
            self.behavior_state["streak"] = 0
            self.behavior_state["frustration"] = min(1.0, self.behavior_state["frustration"] + 0.1)
            if self.behavior_state["frustration"] > 0.7:
                self.behavior_state["motivation"] = max(0.0, self.behavior_state["motivation"] - 0.1)
            self._worsen_posture()
            return False

    def _simulate_user_response(self, action: int) -> bool:
        base_compliance = self.profile.compliance_rate
        
        base_compliance *= (1 + self.behavior_state["motivation"] * 0.2)
        base_compliance *= (1 - self.behavior_state["frustration"] * 0.3)
        
        if self.consecutive_alerts > self.profile.fatigue_threshold:
            fatigue_multiplier = 1 - (self.consecutive_alerts - self.profile.fatigue_threshold) * self.profile.stubbornness
            base_compliance *= max(0.1, fatigue_multiplier)
        
        if action == Action.STRONG_ALERT.value:
            base_compliance += 0.2
        elif action == Action.SUBTLE_ALERT.value:
            base_compliance += 0.05
        
        if self.consecutive_alerts == 1:
            base_compliance *= self.profile.attention_span
        elif self.consecutive_alerts > 1:
            base_compliance *= (1 - 0.1 * (self.consecutive_alerts - 1))
        
        if self.profile.posture_preference == "slouching":
            base_compliance *= 0.7
        elif self.profile.posture_preference == "leaning":
            base_compliance *= 0.75
        
        base_compliance = max(0.05, min(0.98, base_compliance))
        return random.random() < base_compliance

    def _improve_posture(self):
        self.consecutive_alerts = 0
        improvement = 0.15 + self.profile.learning_rate * 0.1
        
        if self.current_posture == PostureLabel.GOOD:
            self.current_score = min(1.0, self.current_score + 0.05)
        elif self.current_posture == PostureLabel.SLOUCHING:
            self.current_posture = PostureLabel.GOOD
            self.current_score = min(1.0, self.current_score + improvement)
        elif self.current_posture == PostureLabel.FORWARD_HEAD:
            self.current_posture = PostureLabel.GOOD
            self.current_score = min(1.0, self.current_score + improvement * 1.2)
        elif self.current_posture == PostureLabel.LEANING:
            self.current_posture = PostureLabel.GOOD
            self.current_score = min(1.0, self.current_score + improvement * 1.2)
        else:
            self.current_posture = PostureLabel.GOOD
            self.current_score = 0.8

    def _worsen_posture(self):
        score_decrease = 0.08 + self.profile.stubbornness * 0.15
        score_decrease *= (1 + self.behavior_state["frustration"] * 0.5)
        self.current_score = max(0.1, self.current_score - score_decrease)
        
        if random.random() < 0.35 + self.profile.stubbornness * 0.2:
            if self.current_posture == PostureLabel.GOOD:
                self.current_posture = random.choice([PostureLabel.SLOUCHING, PostureLabel.FORWARD_HEAD])
            elif self.current_posture == PostureLabel.SLOUCHING and random.random() < 0.5:
                self.current_posture = PostureLabel.FORWARD_HEAD
            elif self.current_posture == PostureLabel.FORWARD_HEAD and random.random() < 0.4:
                self.current_posture = PostureLabel.LEANING

    def natural_drift(self):
        self.behavior_state["fatigue_level"] = min(1.0, self.behavior_state["fatigue_level"] + 0.01)
        
        if random.random() < 0.25 * (1 + self.behavior_state["fatigue_level"]):
            self.current_score = max(0.35, self.current_score - 0.03)
            if self.current_score < 0.6 and self.current_posture == PostureLabel.GOOD:
                if random.random() < 0.5:
                    self.current_posture = random.choice([PostureLabel.SLOUCHING, PostureLabel.FORWARD_HEAD, PostureLabel.LEANING])
                    self.behavior_state["streak"] = 0
        
        if random.random() < self.profile.reset_probability * 0.1:
            self.behavior_state["motivation"] = min(1.0, self.behavior_state["motivation"] + 0.1)
            self.behavior_state["frustration"] = max(0.0, self.behavior_state["frustration"] - 0.05)

    def get_behavior_features(self) -> dict:
        return {
            "fatigue_level": self.behavior_state["fatigue_level"],
            "motivation": self.behavior_state["motivation"],
            "frustration": self.behavior_state["frustration"],
            "streak": self.behavior_state["streak"],
            "best_streak": self.behavior_state["best_streak"],
            "compliance_score": self.total_corrections / max(1, self.total_alerts_received),
        }

    def reset(self):
        self.current_posture = PostureLabel.GOOD
        self.current_score = 0.8
        self.consecutive_alerts = 0
        self.total_alerts_received = 0
        self.total_corrections = 0
        self.successful_alerts = 0
        self.ignored_alerts = 0
        self.alert_history = []
        self.behavior_state["streak"] = 0
        self.behavior_state["fatigue_level"] = max(0, self.behavior_state["fatigue_level"] - 0.2)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "profile": self.profile.to_dict(),
            "behavior_state": self.behavior_state.copy(),
            "total_sessions": len(self.session_history),
        }


class DomainRandomizer:
    def __init__(self, randomization_strength: float = 0.3):
        self.randomization_strength = randomization_strength
        self.perturbation_ranges = {
            "reward_scale": (-0.2, 0.2),
            "threshold_scale": (-0.25, 0.25),
            "delay_scale": (-0.3, 0.3),
            "noise_std": (0.0, 0.1),
        }
        self.current_domain = {}

    def sample_domain(self) -> dict:
        domain = {}
        for param, (low, high) in self.perturbation_ranges.items():
            if random.random() < self.randomization_strength:
                domain[param] = random.uniform(low, high)
            else:
                domain[param] = 0.0
        self.current_domain = domain
        return domain

    def apply_noise(self, state_array: np.ndarray) -> np.ndarray:
        if self.current_domain.get("noise_std", 0) > 0:
            noise = np.random.normal(0, self.current_domain["noise_std"], state_array.shape)
            return np.clip(state_array + noise, 0, 1)
        return state_array


class CurriculumManager:
    def __init__(self, num_stages: int = 3):
        self.num_stages = num_stages
        self.current_stage = 0
        self.episodes_per_stage = 150
        
        self.stages = [
            {"difficulty": "easy", "max_steps": 300, "alert_prob": 0.15, "description": "Easy - Forgiving users"},
            {"difficulty": "medium", "max_steps": 500, "alert_prob": 0.25, "description": "Medium - Standard users"},
            {"difficulty": "hard", "max_steps": 500, "alert_prob": 0.35, "description": "Hard - Stubborn users"},
        ]
        
        self.transition_criteria = {
            0: {"min_reward": 50, "min_episodes": 100},
            1: {"min_reward": 30, "min_episodes": 100},
        }

    def get_current_stage(self) -> dict:
        return self.stages[self.current_stage]

    def should_advance(self, recent_rewards: List[float]) -> bool:
        if self.current_stage >= self.num_stages - 1:
            return False
        
        criteria = self.transition_criteria.get(self.current_stage, {})
        min_episodes = criteria.get("min_episodes", 100)
        min_reward = criteria.get("min_reward", 30)
        
        if len(recent_rewards) < min_episodes:
            return False
        
        avg_reward = np.mean(recent_rewards[-min_episodes:])
        return avg_reward >= min_reward

    def advance_stage(self) -> bool:
        if self.current_stage < self.num_stages - 1:
            self.current_stage += 1
            return True
        return False

    def get_progress(self) -> dict:
        progress = self.current_stage / max(1, self.num_stages - 1)
        return {
            "stage": self.current_stage,
            "stage_name": self.stages[self.current_stage]["description"],
            "progress": progress,
            "total_stages": self.num_stages,
        }


class TrainingSimulator:
    def __init__(self, num_users: int = 5, enable_curriculum: bool = True, 
                 enable_domain_randomization: bool = True):
        self.num_users = num_users
        self.enable_curriculum = enable_curriculum
        self.enable_domain_randomization = enable_domain_randomization
        
        self.curriculum = CurriculumManager() if enable_curriculum else None
        self.domain_randomizer = DomainRandomizer() if enable_domain_randomization else None
        
        self.users = [SimulatedUser(UserBehaviorProfile.random_profile("medium")) for _ in range(num_users)]
        self.current_user_idx: int = 0
        self.episode_count: int = 0

    def get_current_user(self) -> SimulatedUser:
        return self.users[self.current_user_idx]

    def switch_user(self):
        difficulty = "easy"
        if self.curriculum:
            difficulty = self.curriculum.get_current_stage()["difficulty"]
        
        self.current_user_idx = (self.current_user_idx + 1) % len(self.users)
        
        if random.random() < 0.3:
            new_profile = UserBehaviorProfile.random_profile(difficulty)
            self.users[self.current_user_idx] = SimulatedUser(new_profile)
        
        return self.get_current_user()

    def reset_current_user(self):
        self.get_current_user().reset()

    def simulate_step(self, action: int) -> Tuple[PostureLabel, float]:
        user = self.get_current_user()
        if action != Action.NO_FEEDBACK.value:
            user.receive_feedback(action)
        user.natural_drift()
        label, score = user.get_posture()
        return label, score

    def get_user_statistics(self) -> dict:
        user = self.get_current_user()
        compliance_rate = user.total_corrections / user.total_alerts_received if user.total_alerts_received > 0 else 0
        return {
            "user_index": self.current_user_idx,
            "user_id": user.user_id,
            "total_alerts": user.total_alerts_received,
            "total_corrections": user.total_corrections,
            "compliance_rate": compliance_rate,
            "current_posture": user.current_posture.value,
            "current_score": user.current_score,
            "behavior_features": user.get_behavior_features(),
        }

    def get_stage_info(self) -> dict:
        if self.curriculum:
            return self.curriculum.get_progress()
        return {"stage": 0, "stage_name": "Standard", "progress": 1.0}

    def update_curriculum(self, recent_rewards: List[float]) -> dict:
        if self.curriculum and self.curriculum.should_advance(recent_rewards):
            self.curriculum.advance_stage()
            for i, user in enumerate(self.users):
                difficulty = self.curriculum.get_current_stage()["difficulty"]
                new_profile = UserBehaviorProfile.random_profile(difficulty)
                self.users[i].profile = new_profile
        
        return self.get_stage_info()


class UnifiedTrainer:
    def __init__(self, agent, simulator: TrainingSimulator = None, num_episodes: int = 500, 
                 algorithm: str = "ppo", enable_curriculum: bool = True):
        self.agent = agent
        self.simulator = simulator or TrainingSimulator(enable_curriculum=enable_curriculum)
        self.num_episodes = num_episodes
        self.algorithm = algorithm.lower()
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.eval_rewards: List[float] = []
        self.losses: List[float] = []
        self.best_avg_reward: float = float("-inf")
        self.recent_eval_rewards: List[float] = []

    def train(self, eval_interval: int = 50, verbose: bool = True) -> Dict[str, Any]:
        training_stats = {
            "algorithm": self.algorithm,
            "episode_rewards": [],
            "episode_lengths": [],
            "eval_rewards": [],
            "loss_history": [],
            "curriculum_progress": [],
        }
        
        for episode in range(self.num_episodes):
            episode_reward, episode_length, episode_loss = self._train_episode()
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            training_stats["episode_rewards"].append(episode_reward)
            training_stats["episode_lengths"].append(episode_length)
            
            if episode_loss is not None:
                self.losses.append(episode_loss)
                training_stats["loss_history"].append(episode_loss)
            
            if episode % eval_interval == 0:
                eval_reward = self._evaluate(num_episodes=10)
                self.eval_rewards.append(eval_reward)
                self.recent_eval_rewards.append(eval_reward)
                training_stats["eval_rewards"].append(eval_reward)
                
                stage_info = self.simulator.update_curriculum(self.recent_eval_rewards)
                training_stats["curriculum_progress"].append(stage_info)
                
                if eval_reward > self.best_avg_reward:
                    self.best_avg_reward = eval_reward
                    self._save_agent(f"{config.system.model_dir}/{self.algorithm}_best.pth")
                
                if verbose:
                    eps = getattr(self.agent, "epsilon", 0)
                    loss_str = f"{episode_loss:.4f}" if episode_loss is not None else "N/A"
                    stage_str = stage_info.get("stage_name", "N/A")
                    print(f"[{self.algorithm.upper()}] Ep {episode}: Train={episode_reward:.1f}, Eval={eval_reward:.1f}, Stage={stage_str}, Eps={eps:.4f}, Loss={loss_str}")
            
            if episode > 0 and episode % config.system.save_interval == 0:
                self._save_agent(f"{config.system.model_dir}/{self.algorithm}_ep{episode}.pth")
        
        self._save_agent(f"{config.system.model_dir}/{self.algorithm}_final.pth")
        return training_stats

    def _train_episode(self) -> Tuple[float, int, Optional[float]]:
        state = self._reset_episode()
        episode_reward = 0.0
        episode_loss = None
        step = 0
        
        max_steps = 300
        if self.simulator.curriculum:
            max_steps = self.simulator.curriculum.get_current_stage()["max_steps"]
        else:
            max_steps = 500
        
        while step < max_steps:
            state_array = self._get_state_array(state)
            
            if self.simulator.domain_randomizer:
                state_array = self.simulator.domain_randomizer.apply_noise(state_array)
            
            action = self.agent.get_action(state_array, training=True)
            label, score = self.simulator.simulate_step(action)
            next_state, reward, done = self._execute_step(state, action, label, score)
            
            if self.algorithm == "ppo":
                self.agent.record_step(reward, done)
                if done or step >= self.agent.trajectory_size - 1:
                    episode_loss = self._ppo_update()
            else:
                loss = self.agent.update(state_array, action, reward, 
                                        self._get_state_array(next_state), done)
                if loss is not None:
                    episode_loss = loss
            
            episode_reward += reward
            state = next_state
            step += 1
            
            if done:
                break
        
        if self.algorithm == "dqn":
            self.agent.decay_epsilon()
        
        self.simulator.episode_count += 1
        return episode_reward, step, episode_loss

    def _ppo_update(self) -> Optional[float]:
        losses = self.agent.update(0.0)
        return losses.get("loss") if losses else None

    def _reset_episode(self) -> PostureState:
        if random.random() < 0.25:
            self.simulator.switch_user()
        self.simulator.reset_current_user()
        return PostureState(
            posture_label=encode_label(PostureLabel.GOOD),
            posture_score=0.8,
            duration_bad_posture=0.0,
            time_since_alert=0.0,
            recent_corrections=[],
            consecutive_alerts=0,
        )

    def _get_state_array(self, state: PostureState) -> np.ndarray:
        return state.to_array()

    def _execute_step(self, state: PostureState, action: int, label: PostureLabel, 
                      score: float) -> Tuple[PostureState, float, bool]:
        env = PostureEnvironment()
        env.current_state = state
        return env.step(action, label, score)

    def _evaluate(self, num_episodes: int = 10) -> float:
        eval_rewards = []
        for _ in range(num_episodes):
            state = self._reset_episode()
            episode_reward = 0.0
            step = 0
            while step < 200:
                state_array = self._get_state_array(state)
                action = self.agent.get_action(state_array, training=False)
                label, score = self.simulator.simulate_step(action)
                next_state, reward, done = self._execute_step(state, action, label, score)
                episode_reward += reward
                state = next_state
                step += 1
                if done:
                    break
            eval_rewards.append(episode_reward)
        return np.mean(eval_rewards)

    def _save_agent(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.agent.save(path)


class ComparisonBenchmark:
    def __init__(self, simulator: TrainingSimulator):
        self.simulator = simulator

    def benchmark_agent(self, agent, algorithm_name: str, num_episodes: int = 100) -> Dict[str, Any]:
        total_alerts = 0
        successful_corrections = 0
        cumulative_reward = 0.0
        posture_scores = []
        episode_rewards = []
        best_streaks = []
        
        for _ in range(num_episodes):
            self.simulator.reset_current_user()
            state = self._create_initial_state()
            episode_reward = 0.0
            current_streak = 0
            max_streak = 0
            step = 0
            
            while step < 200:
                state_array = state.to_array()
                action = agent.get_action(state_array, training=False)
                if action != Action.NO_FEEDBACK.value:
                    total_alerts += 1
                
                label, score = self.simulator.simulate_step(action)
                env = PostureEnvironment()
                env.current_state = state
                next_state, reward, done = env.step(action, label, score)
                
                was_bad = state.posture_label != PostureLabel.GOOD.value
                is_good = label == PostureLabel.GOOD
                if was_bad and is_good and action != Action.NO_FEEDBACK.value:
                    successful_corrections += 1
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                elif action != Action.NO_FEEDBACK.value:
                    current_streak = 0
                
                episode_reward += reward
                cumulative_reward += reward
                posture_scores.append(score)
                state = next_state
                step += 1
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            best_streaks.append(max_streak)
        
        return {
            "algorithm": algorithm_name,
            "total_alerts": total_alerts,
            "successful_corrections": successful_corrections,
            "correction_rate": successful_corrections / total_alerts if total_alerts > 0 else 0,
            "avg_posture_score": np.mean(posture_scores) if posture_scores else 0,
            "cumulative_reward": cumulative_reward,
            "avg_episode_reward": np.mean(episode_rewards),
            "avg_best_streak": np.mean(best_streaks),
            "total_episodes": num_episodes,
        }

    def benchmark_rule_based(self, num_episodes: int = 100) -> Dict[str, Any]:
        from environment import RuleBasedEnvironment
        env = RuleBasedEnvironment()
        total_alerts = 0
        successful_corrections = 0
        cumulative_reward = 0.0
        posture_scores = []
        episode_rewards = []
        
        for _ in range(num_episodes):
            self.simulator.reset_current_user()
            episode_reward = 0.0
            duration_bad = 0.0
            current_time = 0.0
            step = 0
            while step < 200:
                label, score = self.simulator.simulate_step(Action.NO_FEEDBACK.value)
                if label != PostureLabel.GOOD:
                    duration_bad += config.system.decision_interval
                else:
                    duration_bad = 0.0
                should_alert, action = env.should_alert(label, duration_bad, current_time)
                step_reward = 0.0
                if should_alert:
                    total_alerts += 1
                    was_bad = label != PostureLabel.GOOD
                    _, new_score = self.simulator.simulate_step(action)
                    is_good = new_score >= 0.7 if new_score else False
                    if was_bad and is_good:
                        successful_corrections += 1
                        step_reward = config.reward.posture_improve
                    else:
                        step_reward = config.reward.alert_ignored
                else:
                    if label == PostureLabel.GOOD:
                        step_reward = config.reward.sustained_good
                    elif label != PostureLabel.UNKNOWN:
                        step_reward = config.reward.posture_worsen
                posture_scores.append(score)
                episode_reward += step_reward
                current_time += config.system.decision_interval
                step += 1
            episode_rewards.append(episode_reward)
        
        return {
            "algorithm": "Rule-Based",
            "total_alerts": total_alerts,
            "successful_corrections": successful_corrections,
            "correction_rate": successful_corrections / total_alerts if total_alerts > 0 else 0,
            "avg_posture_score": np.mean(posture_scores) if posture_scores else 0,
            "cumulative_reward": cumulative_reward,
            "avg_episode_reward": np.mean(episode_rewards),
            "total_episodes": num_episodes,
        }

    def compare(self, agents: Dict[str, Any], num_episodes: int = 100) -> Dict[str, Any]:
        results = {}
        for name, agent in agents.items():
            results[name] = self.benchmark_agent(agent, name, num_episodes)
        results["rule_based"] = self.benchmark_rule_based(num_episodes)
        return results

    def _create_initial_state(self) -> PostureState:
        return PostureState(
            posture_label=encode_label(PostureLabel.GOOD),
            posture_score=0.8,
            duration_bad_posture=0.0,
            time_since_alert=0.0,
            recent_corrections=[],
            consecutive_alerts=0,
        )


def train_ppo(num_episodes: int = 500, num_users: int = 5, verbose: bool = True,
               enable_curriculum: bool = True) -> Dict[str, Any]:
    from rl_ppo_agent import PPOPPOAgent
    agent = PPOPPOAgent(state_size=6, action_size=3)
    simulator = TrainingSimulator(num_users=num_users, enable_curriculum=enable_curriculum)
    trainer = UnifiedTrainer(agent, simulator, num_episodes=num_episodes, algorithm="ppo",
                            enable_curriculum=enable_curriculum)
    stats = trainer.train(verbose=verbose)
    return stats


def train_dqn(num_episodes: int = 500, num_users: int = 5, verbose: bool = True,
               enable_curriculum: bool = True) -> Dict[str, Any]:
    from rl_agent import DQNAgent
    agent = DQNAgent(state_size=6, action_size=3)
    simulator = TrainingSimulator(num_users=num_users, enable_curriculum=enable_curriculum)
    trainer = UnifiedTrainer(agent, simulator, num_episodes=num_episodes, algorithm="dqn",
                            enable_curriculum=enable_curriculum)
    stats = trainer.train(verbose=verbose)
    return stats


def compare_algorithms(num_episodes: int = 100, num_users: int = 5, ppo_path: str = None, 
                      dqn_path: str = None) -> Dict[str, Any]:
    from rl_ppo_agent import PPOPPOAgent
    from rl_agent import DQNAgent
    simulator = TrainingSimulator(num_users=num_users, enable_curriculum=False)
    benchmark = ComparisonBenchmark(simulator)
    agents = {}
    ppo_agent = PPOPPOAgent(state_size=6, action_size=3)
    if ppo_path and os.path.exists(ppo_path):
        ppo_agent.load(ppo_path)
    agents["PPO"] = ppo_agent
    dqn_agent = DQNAgent(state_size=6, action_size=3)
    if dqn_path and os.path.exists(dqn_path):
        dqn_agent.load(dqn_path)
    agents["DQN"] = dqn_agent
    results = benchmark.compare(agents, num_episodes)
    return results


if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)
    print("Testing Enhanced Training with Curriculum Learning...")
    print("-" * 60)
    
    print("\nTesting PPO Training (short run with curriculum)...")
    stats = train_ppo(num_episodes=100, num_users=3, verbose=True, enable_curriculum=True)
    print(f"Final eval: {stats['eval_rewards'][-1] if stats['eval_rewards'] else 0:.2f}")
    
    print("\nTesting DQN Training (short run with curriculum)...")
    stats = train_dqn(num_episodes=100, num_users=3, verbose=True, enable_curriculum=True)
    print(f"Final eval: {stats['eval_rewards'][-1] if stats['eval_rewards'] else 0:.2f}")
    
    print("\nComparing algorithms...")
    results = compare_algorithms(num_episodes=50, num_users=3)
    for algo, metrics in results.items():
        print(f"{algo}: Correction Rate={metrics['correction_rate']:.4f}, "
              f"Avg Reward={metrics['avg_episode_reward']:.2f}, "
              f"Avg Streak={metrics.get('avg_best_streak', 0):.1f}")
