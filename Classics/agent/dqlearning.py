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
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = alpha
        self.model = DQN(state_space_size, action_space_size)
        self.target_model = DQN(state_space_size, action_space_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Initialize target model with the same weights as the model
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from the model to the target model."""
        self.target_model.load_state_dict(self.model.state_dict())

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
            return  # not enough experiences in memory to learn

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)  # add dimension for gather
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Get Q values for current states
        current_q_values = self.model(states).gather(1, actions).squeeze(1)
        
        # Compute the expected Q values from the next states using the target network
        next_q_values = self.target_model(next_states).detach().max(1)[0]
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        """
        Decays the exploration rate (epsilon) according to a predefined decay rate.

        This method reduces the value of epsilon by multiplying it by a decay factor,
        ensuring that the exploration rate doesn't fall below a specified minimum value.
        The purpose of decaying epsilon over time is to gradually shift the balance
        from exploration to exploitation as the agent learns about the environment.
        Early in training, a higher epsilon encourages exploration of the state space.
        As learning progresses, reducing epsilon helps the agent to exploit the learned
        policy by choosing actions that it has learned to be effective.

        Parameters:
        None.

        Returns:
        None. The method updates the epsilon attribute of the agent in place.
        """
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        

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
