import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Tuple, List, Optional
from dataclasses import dataclass
from config import config
import os


@dataclass
class Trajectory:
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]
    log_probs: List[float]
    values: List[float]

    def __len__(self):
        return len(self.states)


class ActorCritic(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int]):
        super(ActorCritic, self).__init__()
        self.action_size = action_size
        actor_layers = []
        prev_size = state_size
        for hidden_size in hidden_sizes:
            actor_layers.append(nn.Linear(prev_size, hidden_size))
            actor_layers.append(nn.Tanh())
            prev_size = hidden_size
        self.actor = nn.Sequential(*actor_layers, nn.Linear(prev_size, action_size))
        critic_layers = []
        prev_size = state_size
        for hidden_size in hidden_sizes:
            critic_layers.append(nn.Linear(prev_size, hidden_size))
            critic_layers.append(nn.Tanh())
            prev_size = hidden_size
        self.critic = nn.Sequential(*critic_layers, nn.Linear(prev_size, 1))
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, x: torch.Tensor) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        action_logits = self.actor(x)
        dist = torch.distributions.Categorical(logits=action_logits)
        value = self.critic(x).squeeze(-1)
        return dist, value

    def get_action(self, x: torch.Tensor, training: bool = True) -> Tuple[int, float, float]:
        dist, value = self.forward(x)
        action = dist.sample() if training else dist.probs.argmax(dim=1)
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()


class PPOPPOAgent:
    def __init__(
        self, state_size: int = None, action_size: int = None, hidden_sizes: List[int] = None,
        lr: float = None, gamma: float = 0.99, gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2, entropy_coef: float = 0.01, value_coef: float = 0.5,
        max_grad_norm: float = 0.5, ppo_epochs: int = 5, batch_size: int = None,  # epochs reduced to 5
        trajectory_size: int = 128,
    ):
        self.state_size = state_size or config.rl.state_size
        self.action_size = action_size or config.rl.action_size
        self.hidden_sizes = hidden_sizes or config.rl.hidden_sizes
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs  # Reduced from 10 to 5
        self.batch_size = batch_size or config.rl.batch_size
        self.trajectory_size = trajectory_size
        
        # LR scheduling
        self.initial_lr = lr or config.rl.learning_rate
        self.lr_decay = config.rl.lr_decay
        self.lr_decay_interval = config.rl.lr_decay_interval
        self.total_updates = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(self.state_size, self.action_size, self.hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.initial_lr)
        self.trajectory = Trajectory([], [], [], [], [], [])
        self.trajectory_count = 0
        self.action_names = ["no_feedback", "subtle_alert", "strong_alert"]

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_tensor, training)
        if training:
            self.trajectory.states.append(state.copy())
            self.trajectory.actions.append(action)
            self.trajectory.log_probs.append(log_prob)
            self.trajectory.values.append(value)
        return action

    def record_step(self, reward: float, done: bool):
        self.trajectory.rewards.append(reward)
        self.trajectory.dones.append(done)
        if done or len(self.trajectory) >= self.trajectory_size:
            self.trajectory_count += 1
            return True
        return False

    def compute_gae(self, next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        rewards = self.trajectory.rewards
        dones = self.trajectory.dones
        values = self.trajectory.values + [next_value]
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, next_value: float = 0.0) -> dict:
        # LR scheduling: decay every lr_decay_interval updates
        self.total_updates += 1
        if self.total_updates > 0 and self.total_updates % self.lr_decay_interval == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * self.lr_decay, 1e-5)
        
        if len(self.trajectory) < self.batch_size:
            self._clear_trajectory()
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        advantages, returns = self.compute_gae(next_value)
        states = torch.FloatTensor(np.array(self.trajectory.states)).to(self.device)
        actions = torch.LongTensor(self.trajectory.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.trajectory.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        indices = np.arange(len(self.trajectory))
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                end = min(start + self.batch_size, len(indices))
                batch_indices = indices[start:end]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                dist, values_pred = self.policy(batch_states)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred, batch_returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        avg_losses = {
            "loss": (total_policy_loss + self.value_coef * total_value_loss) / max(num_updates, 1),
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }
        self._clear_trajectory()
        return avg_losses

    def _clear_trajectory(self):
        self.trajectory = Trajectory([], [], [], [], [], [])

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({"policy_state_dict": self.policy.state_dict(), "optimizer_state_dict": self.optimizer.state_dict()}, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return True

    def get_action_name(self, action: int) -> str:
        return self.action_names[action] if action < len(self.action_names) else "unknown"

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, _ = self.policy(state_tensor)
            return dist.probs.cpu().numpy()[0]


if __name__ == "__main__":
    agent = PPOPPOAgent(state_size=18, action_size=3)
    state = np.random.rand(6).astype(np.float32)
    action = agent.get_action(state, training=True)
    print(f"Action: {action} ({agent.get_action_name(action)})")
    agent.record_step(1.0, False)
    agent.record_step(0.5, False)
    agent.record_step(2.0, True)
    losses = agent.update(0.0)
    print(f"Losses: {losses}")
    agent.save("models/ppo_test.pth")
    agent.load("models/ppo_test.pth")
    print("PPO model save/load successful")
