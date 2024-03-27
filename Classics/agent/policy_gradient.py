import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque



class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, action_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
    
class PolicyGradientAgent():
    def __init__(self, state_size, action_space_size):
        super().__init__(action_space_size)
        self.policy_network = PolicyNetwork(state_size, action_space_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=1e-2)
        self.saved_log_probs = []  # to save log probabilities of actions taken
        self.rewards = []  # to save rewards at each step

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network(state)
        action = torch.multinomial(probs, 1).item()
        self.saved_log_probs.append(torch.log(probs[0, action]))
        return action

    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + 0.99 * R  # discount factor gamma=0.99
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # normalize
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.saved_log_probs = []  # reset memories after updating
        self.rewards = []

    def update_q_table(self, state, action, reward, next_state):
        pass  # Not applicable for Policy Gradient

    def get_state(self, game):
        pass  # Implementation depends on the game's state representation