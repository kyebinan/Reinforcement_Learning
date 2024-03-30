import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, action_size)
        self.softmax = nn.Softmax(dim=-1)  # Use dim=-1 for compatibility with both batched and unbatched states

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

class PolicyGradientAgent():
    def __init__(self, state_size, action_size, learning_rate=1e-2):
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.saved_log_probs = []  # To save log probabilities of actions taken
        self.rewards = []  # To save rewards at each step

    def choose_action(self, state):
        # Convert state to tensor if not already, allow for both numpy arrays and tensors
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) if not isinstance(state, torch.Tensor) else state
        probs = self.policy_network(state)
        # Sample action from the action probability distribution
        action = torch.multinomial(probs, 1).item()
        # Save log probability of the chosen action for learning
        self.saved_log_probs.append(torch.log(probs.squeeze(0)[action]))
        return action

    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []  # Store calculated returns for each timestep
        # Calculate discounted return in reverse order
        for r in self.rewards[::-1]:
            R = r + 0.99 * R  # Apply discount factor (gamma)
            returns.insert(0, R)
        # Normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        # Calculate policy gradient loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        # Perform backpropagation
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        # Reset the memory after updating
        self.saved_log_probs = []
        self.rewards = []

    def save_model(self, path):
        torch.save(self.policy_network.state_dict(), path)

    def load_model(self, path):
        self.policy_network.load_state_dict(torch.load(path))
        self.policy_network.eval()
