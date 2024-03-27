import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        self.common = nn.Linear(state_size, 128)
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.relu(self.common(state))
        
        action_probs = self.softmax(self.actor(x))
        state_values = self.critic(x)
        
        return action_probs, state_values
    
class ActorCriticAgent():
    def __init__(self, state_size, action_space_size):
        super().__init__(action_space_size)
        self.model = ActorCritic(state_size, action_space_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)
        
        # Use these to store samples and rewards
        self.saved_actions = []
        self.rewards = []

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, _ = self.model(state)
        action = probs.multinomial(num_samples=1).detach()
        self.saved_actions.append(action)
        return action.item()

    def finish_episode(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            
            # Calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)
            
            # Calculate critic (value) loss using L1 smooth loss
            value_losses.append(nn.functional.smooth_l1_loss(value, torch.tensor([[R]])))
        
        self.optimizer.zero_grad()
        
        # Sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        
        # Perform backprop
        loss.backward()
        self.optimizer.step()
        
        # Reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

    def update_q_table(self, state, action, reward, next_state):
        pass  # Not applicable for Actor-Critic

    def get_state(self, game):
        pass  # Implementation depends on the game's state representation