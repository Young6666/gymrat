import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

LEARNING_RATE = 0.0003
GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.2
K_EPOCH = 10
HIDDEN_DIM = 64

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, 1)
        )
        
        # Actor Network
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, action_dim),
            nn.Tanh() # Action: -1 ~ 1
        )
        
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, state):
        return self.critic(state)

    def get_action(self, state):
        mu = self.actor_mean(state)
        sigma = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        return action, action_log_prob

    def evaluate(self, state, action):
        mu = self.actor_mean(state)
        sigma = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(mu, sigma)
        
        action_log_probs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        
        return action_log_probs, state_values, dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer = [] 
    
    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, action_log_prob = self.policy_old.get_action(state_tensor)
            
        action = action.cpu().numpy().flatten()
        
        return action, action_log_prob.item()

    def store_transition(self, transition):
        self.buffer.append(transition)

    def update(self):
        states = torch.tensor(np.array([t[0] for t in self.buffer]), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array([t[1] for t in self.buffer]), dtype=torch.float).to(self.device)
        rewards = [t[2] for t in self.buffer]
        dones = [t[4] for t in self.buffer]
        old_log_probs = torch.tensor(np.array([t[5] for t in self.buffer]), dtype=torch.float).to(self.device)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            values = self.policy.get_value(states).squeeze()
            
        returns = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = 0
            else:
                next_val = values[i + 1]
            
            delta = rewards[i] + GAMMA * next_val * (1 - dones[i]) - values[i]
            gae = delta + GAMMA * LAMBDA * (1 - dones[i]) * gae
            returns.insert(0, gae + values[i])
            
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        for _ in range(K_EPOCH):
            log_probs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            state_values = state_values.squeeze()
            
            ratios = torch.exp(log_probs - old_log_probs)
            
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, returns) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = []