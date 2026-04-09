import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass, field
import json
import os
import time
from config import config


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: float
    posture_label: str
    posture_score: float
    user_id: str = "default"

    def to_dict(self) -> dict:
        return {
            "state": self.state.tolist(),
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state.tolist(),
            "done": self.done,
            "timestamp": self.timestamp,
            "posture_label": self.posture_label,
            "posture_score": self.posture_score,
            "user_id": self.user_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Experience":
        return cls(
            state=np.array(data["state"], dtype=np.float32),
            action=data["action"],
            reward=data["reward"],
            next_state=np.array(data["next_state"], dtype=np.float32),
            done=data["done"],
            timestamp=data["timestamp"],
            posture_label=data["posture_label"],
            posture_score=data["posture_score"],
            user_id=data.get("user_id", "default"),
        )


@dataclass
class OnlineLearningConfig:
    enabled: bool = True
    update_frequency: int = 50
    batch_size: int = 16
    learning_rate: float = 0.0001
    store_experience: bool = True
    max_buffer_size: int = 5000
    min_experiences: int = 32
    importance_sampling: bool = True
    behavioral_cloning_weight: float = 0.1
    save_interval: int = 300
    auto_adjust_lr: bool = True
    convergence_threshold: float = 0.01
    performance_window: int = 100


class ExperienceBuffer:
    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self.buffer: deque[Experience] = deque(maxlen=max_size)
        self.priorities: deque[float] = deque(maxlen=max_size)
        self.episodic_buffer: List[Experience] = []

    def add(self, experience: Experience, priority: float = 1.0):
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.episodic_buffer.append(experience)

    def clear_episodic(self):
        self.episodic_buffer = []

    def sample(self, batch_size: int, use_priorities: bool = True) -> List[Experience]:
        if len(self.buffer) == 0:
            return []
        
        if not use_priorities or len(set(self.priorities)) <= 1:
            indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        else:
            probs = np.array(self.priorities) ** 2
            probs /= probs.sum()
            indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), 
                                      p=probs, replace=False)
        
        return [self.buffer[i] for i in indices]

    def update_priorities(self, indices: List[int], td_errors: List[float]):
        for idx, error in zip(indices, td_errors):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = abs(error) + 1e-5

    def get_recent_experiences(self, n: int = 100) -> List[Experience]:
        return list(self.buffer)[-n:]

    def __len__(self) -> int:
        return len(self.buffer)


class OnlineLearner:
    def __init__(self, agent, config: OnlineLearningConfig = None):
        self.agent = agent
        self.config = config or OnlineLearningConfig()
        self.buffer = ExperienceBuffer(max_size=self.config.max_buffer_size)
        self.step_count = 0
        self.update_count = 0
        self.last_save_time = time.time()
        
        self.performance_history: deque[float] = deque(maxlen=self.config.performance_window)
        self.loss_history: deque[float] = deque(maxlen=100)
        self.lr_history: deque[float] = deque(maxlen=100)
        
        self.current_lr = self.config.learning_rate
        self.optimizer = None
        if hasattr(agent, 'optimizer') and agent.optimizer is not None:
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = self.current_lr
            self.optimizer = agent.optimizer
        
        self.episode_count = 0
        self.episode_rewards: List[float] = []
        self.is_learning = False

    def add_experience(self, state: np.ndarray, action: int, reward: float,
                       next_state: np.ndarray, done: bool,
                       posture_label: str = "unknown", posture_score: float = 0.0,
                       user_id: str = "default"):
        if not self.config.enabled or not self.config.store_experience:
            return
        
        experience = Experience(
            state=state.copy(),
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            timestamp=time.time(),
            posture_label=posture_label,
            posture_score=posture_score,
            user_id=user_id,
        )
        
        priority = self._compute_priority(experience)
        self.buffer.add(experience, priority)
        
        self.step_count += 1
        
        if done:
            self.episode_count += 1
            total_reward = sum(e.reward for e in self.buffer.episodic_buffer)
            self.episode_rewards.append(total_reward)
            self.performance_history.append(total_reward)
            self.buffer.clear_episodic()
            
            if len(self.performance_history) >= 10:
                self._maybe_adjust_learning_rate()
        
        if self.step_count % self.config.update_frequency == 0:
            self.update()

    def _compute_priority(self, experience: Experience) -> float:
        base_priority = 1.0
        
        if experience.reward > 0:
            base_priority *= 1.5
        
        if experience.action != 0:
            base_priority *= 1.2
        
        if abs(experience.reward) > 5:
            base_priority *= 1.5
        
        return base_priority

    def update(self) -> Optional[float]:
        if not self.config.enabled:
            return None
        
        if len(self.buffer) < self.config.min_experiences:
            return None
        
        if not hasattr(self.agent, 'update_online'):
            return self._basic_update()
        
        experiences = self.buffer.sample(self.config.batch_size)
        if not experiences:
            return None
        
        states = np.array([e.state for e in experiences], dtype=np.float32)
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences], dtype=np.float32)
        dones = np.array([e.done for e in experiences])
        
        loss = self.agent.update_online(states, actions, rewards, next_states, dones,
                                        learning_rate=self.current_lr)
        
        if loss is not None:
            self.loss_history.append(loss)
            self.update_count += 1
            
            if self.config.save_interval > 0:
                current_time = time.time()
                if current_time - self.last_save_time > self.config.save_interval:
                    self._save_checkpoint()
                    self.last_save_time = current_time
        
        return loss

    def _basic_update(self) -> Optional[float]:
        if self.optimizer is None or not hasattr(self.agent, 'q_network'):
            return None
        
        experiences = self.buffer.sample(self.config.batch_size)
        if not experiences:
            return None
        
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.agent.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.agent.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.agent.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.agent.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.agent.device)
        
        current_q = self.agent.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.agent.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.agent.gamma * next_q
        
        loss = torch.nn.functional.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        self.update_count += 1
        
        return loss.item()

    def _maybe_adjust_learning_rate(self):
        if not self.config.auto_adjust_lr or len(self.loss_history) < 50:
            return
        
        recent_losses = list(self.loss_history)[-50:]
        early_avg = np.mean(recent_losses[:25])
        late_avg = np.mean(recent_losses[-25:])
        
        if early_avg > 0.01:
            loss_change = abs(late_avg - early_avg) / early_avg
        else:
            loss_change = 0
        
        if loss_change < self.config.convergence_threshold:
            if len(self.performance_history) >= 50:
                recent_perf = np.mean(list(self.performance_history)[-50:])
                if recent_perf < np.mean(list(self.performance_history)[-100:-50]) if len(self.performance_history) >= 100 else False:
                    self.current_lr = max(self.current_lr * 0.5, 1e-6)
                    self._update_learning_rate()
        elif loss_change > 0.5:
            self.current_lr = min(self.current_lr * 1.1, self.config.learning_rate * 2)
            self._update_learning_rate()
        
        self.lr_history.append(self.current_lr)

    def _update_learning_rate(self):
        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
        elif hasattr(self.agent, 'optimizer') and self.agent.optimizer:
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = self.current_lr

    def _save_checkpoint(self):
        checkpoint_path = f"{config.system.model_dir}/online_learning_checkpoint.pt"
        os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".", exist_ok=True)
        
        checkpoint = {
            "step_count": self.step_count,
            "update_count": self.update_count,
            "episode_count": self.episode_count,
            "current_lr": self.current_lr,
            "loss_history": list(self.loss_history),
            "performance_history": list(self.performance_history),
            "agent_state": None,
        }
        
        if hasattr(self.agent, 'save'):
            agent_path = f"{config.system.model_dir}/online_agent_temp.pth"
            self.agent.save(agent_path)
            checkpoint["agent_state"] = agent_path
        
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str = None) -> bool:
        if checkpoint_path is None:
            checkpoint_path = f"{config.system.model_dir}/online_learning_checkpoint.pt"
        
        if not os.path.exists(checkpoint_path):
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.step_count = checkpoint.get("step_count", 0)
            self.update_count = checkpoint.get("update_count", 0)
            self.episode_count = checkpoint.get("episode_count", 0)
            self.current_lr = checkpoint.get("current_lr", self.config.learning_rate)
            self.loss_history = deque(checkpoint.get("loss_history", []), maxlen=100)
            self.performance_history = deque(checkpoint.get("performance_history", []), 
                                             maxlen=self.config.performance_window)
            
            if checkpoint.get("agent_state"):
                self.agent.load(checkpoint["agent_state"])
            
            return True
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            "enabled": self.config.enabled,
            "buffer_size": len(self.buffer),
            "step_count": self.step_count,
            "update_count": self.update_count,
            "episode_count": self.episode_count,
            "current_lr": self.current_lr,
            "is_learning": self.is_learning,
        }
        
        if len(self.loss_history) > 0:
            stats["avg_loss"] = np.mean(list(self.loss_history))
            stats["recent_loss"] = list(self.loss_history)[-10:]
        
        if len(self.performance_history) > 0:
            stats["avg_episode_reward"] = np.mean(list(self.performance_history))
            stats["recent_performance"] = list(self.performance_history)[-20:]
        
        if len(self.lr_history) > 0:
            stats["lr_trend"] = list(self.lr_history)
        
        return stats

    def export_experiences(self, filepath: str, max_experiences: int = 10000):
        experiences = list(self.buffer)[-max_experiences:]
        data = [exp.to_dict() for exp in experiences]
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        return len(data)

    def import_experiences(self, filepath: str) -> int:
        if not os.path.exists(filepath):
            return 0
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        count = 0
        for exp_data in data:
            try:
                exp = Experience.from_dict(exp_data)
                priority = self._compute_priority(exp)
                self.buffer.add(exp, priority)
                count += 1
            except Exception:
                continue
        
        return count

    def reset(self):
        self.buffer = ExperienceBuffer(max_size=self.config.max_buffer_size)
        self.step_count = 0
        self.update_count = 0
        self.episode_count = 0
        self.current_lr = self.config.learning_rate
        self.performance_history.clear()
        self.loss_history.clear()
        self.lr_history.clear()
        self.episode_rewards = []


class BehavioralCloning:
    def __init__(self, agent):
        self.agent = agent
        self.expert_demonstrations: List[Tuple[np.ndarray, int]] = []

    def add_demonstration(self, state: np.ndarray, action: int):
        self.expert_demonstrations.append((state, action))
        if len(self.expert_demonstrations) > 1000:
            self.expert_demonstrations.pop(0)

    def compute_bc_loss(self, batch_size: int = 32) -> Optional[torch.Tensor]:
        if len(self.expert_demonstrations) < batch_size:
            return None
        
        indices = np.random.choice(len(self.expert_demonstrations), batch_size, replace=False)
        states = torch.FloatTensor(np.array([self.expert_demonstrations[i][0] for i in indices])).to(self.agent.device)
        expert_actions = torch.LongTensor([self.expert_demonstrations[i][1] for i in indices]).to(self.agent.device)
        
        if hasattr(self.agent, 'q_network'):
            predicted_q = self.agent.q_network(states)
            policy_loss = torch.nn.functional.cross_entropy(predicted_q, expert_actions)
        elif hasattr(self.agent, 'policy'):
            dist, _ = self.agent.policy(states)
            policy_loss = -dist.log_prob(expert_actions).mean()
        else:
            return None
        
        return policy_loss

    def apply_bc_loss(self, rl_loss: float, weight: float = 0.1) -> Tuple[float, Optional[float]]:
        bc_loss = self.compute_bc_loss()
        if bc_loss is None:
            return rl_loss, None
        total_loss = rl_loss + weight * bc_loss.item()
        return total_loss, bc_loss.item()


class OfflineReplayLearner:
    def __init__(self, agent):
        self.agent = agent
        self.replay_buffer: List[Experience] = []

    def load_session_log(self, filepath: str) -> int:
        if not os.path.exists(filepath):
            return 0
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        count = 0
        for frame_data in data.get("posture_scores", []):
            if isinstance(frame_data, dict) and "state" in frame_data:
                try:
                    exp = Experience.from_dict(frame_data)
                    self.replay_buffer.append(exp)
                    count += 1
                except Exception:
                    continue
        
        return count

    def train_from_offline_data(self, epochs: int = 10, batch_size: int = 64,
                               learning_rate: float = 0.001) -> Dict[str, List[float]]:
        if len(self.replay_buffer) < batch_size:
            return {"losses": [], "rewards": []}
        
        losses = []
        rewards = []
        
        optimizer = torch.optim.Adam(self.agent.q_network.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            batch_indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in batch_indices]
            
            states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.agent.device)
            actions = torch.LongTensor([e.action for e in batch]).to(self.agent.device)
            batch_rewards = torch.FloatTensor([e.reward for e in batch]).to(self.agent.device)
            next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.agent.device)
            dones = torch.FloatTensor([e.done for e in batch]).to(self.agent.device)
            
            current_q = self.agent.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = self.agent.target_network(next_states).max(1)[0]
                target_q = batch_rewards + (1 - dones) * self.agent.gamma * next_q
            
            loss = torch.nn.functional.mse_loss(current_q, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
            rewards.append(batch_rewards.mean().item())
        
        if epochs % 10 == 0:
            self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())
        
        return {"losses": losses, "rewards": rewards}


def create_online_learner(agent, config: OnlineLearningConfig = None) -> OnlineLearner:
    return OnlineLearner(agent, config)


if __name__ == "__main__":
    print("Testing Online Learning Module...")
    
    from rl_agent import DQNAgent
    
    agent = DQNAgent(state_size=6, action_size=3)
    learner = OnlineLearner(agent)
    
    print(f"Initial stats: {learner.get_statistics()}")
    
    for episode in range(5):
        state = np.random.rand(6).astype(np.float32)
        for step in range(100):
            action = np.random.randint(0, 3)
            next_state = np.random.rand(6).astype(np.float32)
            reward = np.random.uniform(-5, 10)
            done = step == 99
            
            learner.add_experience(state, action, reward, next_state, done)
            state = next_state
    
    print(f"After adding experiences: {learner.get_statistics()}")
    
    for _ in range(10):
        loss = learner.update()
        if loss:
            print(f"Update loss: {loss:.4f}")
    
    print(f"Final stats: {learner.get_statistics()}")
    
    print("\nTesting offline learning...")
    learner.export_experiences("test_experiences.json")
    
    new_learner = OnlineLearner(agent)
    count = new_learner.import_experiences("test_experiences.json")
    print(f"Imported {count} experiences")
