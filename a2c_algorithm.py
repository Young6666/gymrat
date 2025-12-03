import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

LEARNING_RATE = 0.0005
GAMMA = 0.99
ENTROPY_BETA = 0.01
HIDDEN_DIM = 64

class A2CNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A2CNetwork, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )
        
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, action_dim),
            nn.Tanh()
        )
        
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        value = self.critic(state)
        mu = self.actor_mean(state)
        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(mu, std)
        return dist, value

class A2CAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = A2CNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        self.buffer = [] 

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, _ = self.network(state_tensor)
            action = dist.sample()
            
        action = action.cpu().numpy().flatten()
        return action, None

    def store_transition(self, transition):
        self.buffer.append(transition)

    def update(self):
        if not self.buffer:
            return

        states = torch.tensor(np.array([t[0] for t in self.buffer]), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array([t[1] for t in self.buffer]), dtype=torch.float).to(self.device)
        rewards = [t[2] for t in self.buffer]
        dones = [t[4] for t in self.buffer]
        
        returns = []
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + GAMMA * R * (1 - dones[i])
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        dist, values = self.network(states)
        values = values.squeeze()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = nn.MSELoss()(values, returns)
        
        loss = actor_loss + 0.5 * critic_loss - ENTROPY_BETA * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        self.buffer = []