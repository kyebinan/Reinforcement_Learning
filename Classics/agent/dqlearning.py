import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self, state_space_size, action_space_size):
        super(DQN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(state_space_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 24)
        self.fc4 = nn.Linear(24, action_space_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    def __init__(self, state_space_size, action_space_size, alpha=0.001, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_space_size
        self.action_size = action_space_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma   # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = alpha
        self.model = DQN(state_space_size, action_space_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        # print(f"State shape before processing: {state.shape}")
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension
        # print(f"Processed state tensor shape: {state.shape}")  # Debugging print
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        Q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_Q_values = self.model(next_states).detach().max(1)[0]
        targets = rewards + self.gamma * next_Q_values * (1 - dones)

        loss = nn.SmoothL1Loss()(Q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename='dqn_model.pth'):
       """
        Saves the model's state dictionary to a file.

        Parameters:
        - filename (str): The path to the file where the model's state dictionary should be saved.
        """
       torch.save(self.model.state_dict(), filename)


    def load(self, filename='dqn_model.pth'):
        """
        Loads the model's state dictionary from a file.

        Parameters:
        - filename (str): The path to the file from which the model's state dictionary should be loaded.
        """
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()  # Set the model to evaluation mode
