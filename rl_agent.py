import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Optional
from config import config
import os


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int]):
        super(QNetwork, self).__init__()
        layers = []
        prev_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self, state_size: int = None, action_size: int = None, hidden_sizes: List[int] = None,
        gamma: float = None, learning_rate: float = None, batch_size: int = None,
        target_update_freq: int = None, learning_freq: int = None, gradient_clip: float = None,
        epsilon_start: float = None, epsilon_min: float = None, epsilon_decay: float = None,
        memory_size: int = None,
    ):
        self.state_size = state_size or config.rl.state_size
        self.action_size = action_size or config.rl.action_size
        self.hidden_sizes = hidden_sizes or config.rl.hidden_sizes
        self.gamma = gamma if gamma is not None else config.rl.gamma
        self.learning_rate = learning_rate if learning_rate is not None else config.rl.learning_rate
        self.batch_size = batch_size if batch_size is not None else config.rl.batch_size
        self.target_update_freq = target_update_freq or config.rl.target_update_freq
        self.learning_freq = learning_freq or config.rl.learning_freq
        self.gradient_clip = gradient_clip if gradient_clip is not None else config.rl.gradient_clip
        self.epsilon = epsilon_start if epsilon_start is not None else config.rl.epsilon_start
        self.epsilon_min = epsilon_min if epsilon_min is not None else config.rl.epsilon_min
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else config.rl.epsilon_decay
        self.memory = ReplayBuffer(memory_size or config.rl.memory_size)
        self.learn_step = 0
        
        # LR scheduling
        self.lr_decay = config.rl.lr_decay
        self.lr_decay_interval = config.rl.lr_decay_interval
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(self.state_size, self.action_size, self.hidden_sizes).to(self.device)
        self.target_network = QNetwork(self.state_size, self.action_size, self.hidden_sizes).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.action_names = ["no_feedback", "subtle_alert", "strong_alert"]

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def get_action_name(self, action: int) -> str:
        return self.action_names[action]

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> Optional[float]:
        self.memory.push(Transition(state, action, reward, next_state, done))
        if len(self.memory) < self.batch_size or self.learn_step % self.learning_freq != 0:
            self.learn_step += 1
            return None
        
        # LR scheduling: decay every lr_decay_interval updates
        if self.learn_step > 0 and self.learn_step % self.lr_decay_interval == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * self.lr_decay, 1e-5)
        
        batch = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor([t.done for t in batch]).to(self.device)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_step": self.learn_step,
        }, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.learn_step = checkpoint["learn_step"]
        return True

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_network(state_tensor).cpu().numpy()[0]


def create_agent(agent_type: str = "dqn", state_size: int = None, action_size: int = None):
    if agent_type.lower() == "dqn":
        return DQNAgent(state_size, action_size)
    raise ValueError(f"Unknown agent type: {agent_type}. Available: dqn")


if __name__ == "__main__":
    agent = DQNAgent(state_size=18, action_size=3)
    state = np.random.rand(6).astype(np.float32)
    action = agent.get_action(state)
    print(f"State shape: {state.shape}, Action: {action} ({agent.get_action_name(action)})")
    next_state = np.random.rand(6).astype(np.float32)
    for _ in range(100):
        agent.update(state, action, 1.0, next_state, False)
    loss = agent.update(state, action, 1.0, next_state, False)
    print(f"Training loss: {loss}")
    agent.save("models/test_dqn.pth")
    agent.load("models/test_dqn.pth")
    print("DQN model saved and loaded successfully")