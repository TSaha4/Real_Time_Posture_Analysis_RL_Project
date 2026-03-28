import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from posture_module import PostureLabel, encode_label
from environment import Action, PostureState, PostureEnvironment
from config import config
import random
import os


@dataclass
class UserBehaviorProfile:
    compliance_rate: float = 0.7
    fatigue_threshold: int = 3
    stubbornness: float = 0.2
    attention_span: float = 0.8

    @classmethod
    def random_profile(cls) -> "UserBehaviorProfile":
        return cls(
            compliance_rate=random.uniform(0.5, 0.9),
            fatigue_threshold=random.randint(2, 5),
            stubbornness=random.uniform(0.1, 0.4),
            attention_span=random.uniform(0.6, 1.0),
        )


class SimulatedUser:
    def __init__(self, profile: UserBehaviorProfile = None):
        self.profile = profile or UserBehaviorProfile()
        self.current_posture: PostureLabel = PostureLabel.GOOD
        self.current_score: float = 0.8
        self.consecutive_alerts: int = 0
        self.alert_history: list = []
        self.total_alerts_received: int = 0
        self.total_corrections: int = 0

    def get_posture(self) -> Tuple[PostureLabel, float]:
        return self.current_posture, self.current_score

    def receive_feedback(self, action: int) -> bool:
        if action == Action.NO_FEEDBACK.value:
            return False
        self.total_alerts_received += 1
        self.consecutive_alerts += 1
        self.alert_history.append(action)
        correction_made = self._simulate_user_response(action)
        if correction_made:
            self.total_corrections += 1
            self._improve_posture()
            return True
        self._worsen_posture()
        return False

    def _simulate_user_response(self, action: int) -> bool:
        base_compliance = self.profile.compliance_rate
        if self.consecutive_alerts > self.profile.fatigue_threshold:
            base_compliance *= (1 - self.profile.stubbornness * self.consecutive_alerts)
        if action == Action.STRONG_ALERT.value:
            base_compliance += 0.15
        if self.consecutive_alerts == 1:
            base_compliance *= self.profile.attention_span
        base_compliance = max(0.1, min(0.95, base_compliance))
        return random.random() < base_compliance

    def _improve_posture(self):
        self.consecutive_alerts = 0
        if self.current_posture == PostureLabel.GOOD:
            self.current_score = min(1.0, self.current_score + 0.05)
        elif self.current_posture == PostureLabel.SLOUCHING:
            self.current_posture = PostureLabel.GOOD
            self.current_score = min(1.0, self.current_score + 0.15)
        elif self.current_posture == PostureLabel.FORWARD_HEAD:
            self.current_posture = PostureLabel.GOOD
            self.current_score = min(1.0, self.current_score + 0.2)
        elif self.current_posture == PostureLabel.LEANING:
            self.current_posture = PostureLabel.GOOD
            self.current_score = min(1.0, self.current_score + 0.2)
        else:
            self.current_posture = PostureLabel.GOOD
            self.current_score = 0.8

    def _worsen_posture(self):
        score_decrease = 0.05 + self.profile.stubbornness * 0.1
        self.current_score = max(0.1, self.current_score - score_decrease)
        if random.random() < 0.3:
            if self.current_posture == PostureLabel.GOOD:
                self.current_posture = PostureLabel.SLOUCHING
            elif self.current_posture == PostureLabel.SLOUCHING and random.random() < 0.5:
                self.current_posture = PostureLabel.FORWARD_HEAD
            elif self.current_posture == PostureLabel.FORWARD_HEAD and random.random() < 0.3:
                self.current_posture = PostureLabel.LEANING

    def natural_drift(self):
        if random.random() < 0.2:
            self.current_score = max(0.3, self.current_score - 0.02)
            if self.current_score < 0.5 and self.current_posture == PostureLabel.GOOD and random.random() < 0.4:
                postures = [PostureLabel.SLOUCHING, PostureLabel.FORWARD_HEAD, PostureLabel.LEANING]
                self.current_posture = random.choice(postures)

    def reset(self):
        self.current_posture = PostureLabel.GOOD
        self.current_score = 0.8
        self.consecutive_alerts = 0
        self.total_alerts_received = 0
        self.total_corrections = 0
        self.alert_history = []


class TrainingSimulator:
    def __init__(self, num_users: int = 5):
        self.users = [SimulatedUser(UserBehaviorProfile.random_profile()) for _ in range(num_users)]
        self.current_user_idx: int = 0

    def get_current_user(self) -> SimulatedUser:
        return self.users[self.current_user_idx]

    def switch_user(self):
        self.current_user_idx = (self.current_user_idx + 1) % len(self.users)
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
        return {
            "user_index": self.current_user_idx,
            "total_alerts": user.total_alerts_received,
            "total_corrections": user.total_corrections,
            "compliance_rate": user.total_corrections / user.total_alerts_received if user.total_alerts_received > 0 else 0,
            "current_posture": user.current_posture.value,
            "current_score": user.current_score,
        }


class UnifiedTrainer:
    def __init__(self, agent, simulator: TrainingSimulator = None, num_episodes: int = 500, algorithm: str = "ppo"):
        self.agent = agent
        self.simulator = simulator or TrainingSimulator()
        self.num_episodes = num_episodes
        self.algorithm = algorithm.lower()
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.eval_rewards: List[float] = []
        self.losses: List[float] = []
        self.best_avg_reward: float = float("-inf")

    def train(self, eval_interval: int = 50, verbose: bool = True) -> Dict[str, Any]:
        training_stats = {
            "algorithm": self.algorithm,
            "episode_rewards": [],
            "episode_lengths": [],
            "eval_rewards": [],
            "loss_history": [],
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
                training_stats["eval_rewards"].append(eval_reward)
                if eval_reward > self.best_avg_reward:
                    self.best_avg_reward = eval_reward
                    self._save_agent(f"{config.system.model_dir}/{self.algorithm}_best.pth")
                if verbose:
                    eps = getattr(self.agent, "epsilon", 0)
                    loss_str = f"{episode_loss:.4f}" if episode_loss is not None else "N/A"
                    print(f"[{self.algorithm.upper()}] Episode {episode}: Train={episode_reward:.2f}, Eval={eval_reward:.2f}, Eps={eps:.4f}, Loss={loss_str}")
            if episode > 0 and episode % config.system.save_interval == 0:
                self._save_agent(f"{config.system.model_dir}/{self.algorithm}_ep{episode}.pth")
        self._save_agent(f"{config.system.model_dir}/{self.algorithm}_final.pth")
        return training_stats

    def _train_episode(self) -> Tuple[float, int, Optional[float]]:
        state = self._reset_episode()
        episode_reward = 0.0
        episode_loss = None
        step = 0
        max_steps = 500
        while step < max_steps:
            state_array = self._get_state_array(state)
            action = self.agent.get_action(state_array, training=True)
            label, score = self.simulator.simulate_step(action)
            next_state, reward, done = self._execute_step(state, action, label, score)
            if self.algorithm == "ppo":
                self.agent.record_step(reward, done)
                if done or step >= self.agent.trajectory_size - 1:
                    episode_loss = self._ppo_update()
            else:
                loss = self.agent.update(state_array, action, reward, self._get_state_array(next_state), done)
                if loss is not None:
                    episode_loss = loss
            episode_reward += reward
            state = next_state
            step += 1
            if done:
                break
        if self.algorithm == "dqn":
            self.agent.decay_epsilon()
        return episode_reward, step, episode_loss

    def _ppo_update(self) -> Optional[float]:
        losses = self.agent.update(0.0)
        return losses.get("loss") if losses else None

    def _reset_episode(self) -> PostureState:
        if random.random() < 0.2:
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

    def _execute_step(self, state: PostureState, action: int, label: PostureLabel, score: float) -> Tuple[PostureState, float, bool]:
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
        for _ in range(num_episodes):
            self.simulator.reset_current_user()
            state = self._create_initial_state()
            episode_reward = 0.0
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
                episode_reward += reward
                cumulative_reward += reward
                posture_scores.append(score)
                state = next_state
                step += 1
                if done:
                    break
            episode_rewards.append(episode_reward)
        return {
            "algorithm": algorithm_name,
            "total_alerts": total_alerts,
            "successful_corrections": successful_corrections,
            "correction_rate": successful_corrections / total_alerts if total_alerts > 0 else 0,
            "avg_posture_score": np.mean(posture_scores) if posture_scores else 0,
            "cumulative_reward": cumulative_reward,
            "avg_episode_reward": np.mean(episode_rewards),
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


def train_ppo(num_episodes: int = 500, num_users: int = 5, verbose: bool = True) -> Dict[str, Any]:
    from rl_ppo_agent import PPOPPOAgent
    agent = PPOPPOAgent(state_size=6, action_size=3)
    simulator = TrainingSimulator(num_users=num_users)
    trainer = UnifiedTrainer(agent, simulator, num_episodes=num_episodes, algorithm="ppo")
    stats = trainer.train(verbose=verbose)
    return stats


def train_dqn(num_episodes: int = 500, num_users: int = 5, verbose: bool = True) -> Dict[str, Any]:
    from rl_agent import DQNAgent
    agent = DQNAgent(state_size=6, action_size=3)
    simulator = TrainingSimulator(num_users=num_users)
    trainer = UnifiedTrainer(agent, simulator, num_episodes=num_episodes, algorithm="dqn")
    stats = trainer.train(verbose=verbose)
    return stats


def compare_algorithms(num_episodes: int = 100, num_users: int = 5, ppo_path: str = None, dqn_path: str = None) -> Dict[str, Any]:
    from rl_ppo_agent import PPOPPOAgent
    from rl_agent import DQNAgent
    simulator = TrainingSimulator(num_users=num_users)
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
    print("Testing PPO Training (short run)...")
    stats = train_ppo(num_episodes=50, num_users=3, verbose=True)
    print(f"Final eval: {stats['eval_rewards'][-1] if stats['eval_rewards'] else 0:.2f}")
    print("Testing DQN Training (short run)...")
    stats = train_dqn(num_episodes=50, num_users=3, verbose=True)
    print(f"Final eval: {stats['eval_rewards'][-1] if stats['eval_rewards'] else 0:.2f}")
    print("Comparing algorithms...")
    results = compare_algorithms(num_episodes=50, num_users=3)
    for algo, metrics in results.items():
        print(f"{algo}: Correction Rate={metrics['correction_rate']:.4f}, Avg Reward={metrics['avg_episode_reward']:.2f}")
