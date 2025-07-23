import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, size, num_actions):
        super().__init__()
        self.size = size
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(size*size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

class Agent:
    def __init__(self, size, num_actions, device='cpu'):
        self.size = size
        self.num_actions = num_actions
        self.device = device
        self.policy_net = DQN(size, num_actions).to(device)
        self.target_net = DQN(size, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = []
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, state, valid_mask=None):
        if np.random.rand() < self.epsilon:
            # Chọn ngẫu nhiên trong các action hợp lệ
            if valid_mask is not None:
                valid_indices = np.where(valid_mask)[0]
                return np.random.choice(valid_indices)
            return np.random.randint(self.num_actions)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state).cpu().numpy()[0]
        if valid_mask is not None:
            q_values[~valid_mask] = -np.inf
        return int(np.argmax(q_values))

    def store(self, transition):
        self.memory.append(transition)
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def sample(self):
        idx = np.random.choice(len(self.memory), self.batch_size, replace=False)
        return [self.memory[i] for i in idx]

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.sample()
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict()) 