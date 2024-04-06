import numpy as np
import random


class SARSAAgent():
    def __init__(self, state_space_size, action_space_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01):
        self.action_space_size = action_space_size
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate of epsilon over time
        self.epsilon_min = epsilon_min  # Minimum value of epsilon

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, self.action_space_size - 1)  # Explore
        else:
            action = np.argmax(self.q_table[state])  # Exploit
        return action

    def update_q_table(self, state, action, reward, next_state, next_action):
        """
        SARSA update rule.
        """
        # Predicted (current) Q-value
        q_predict = self.q_table[state, action]
        # Target Q-value
        q_target = reward + self.gamma * self.q_table[next_state, next_action]
        # Update Q-value
        self.q_table[state, action] += self.alpha * (q_target - q_predict)

    
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

    def save_q_table(self, filename='q_table_sarsa.npy'):
        """
        Saves the Q-table to a file.

        Parameters:
        - filename: The name of the file where the Q-table should be saved. The default name is 'q_table.npy'.
        
        Returns:
        None.
        """
        np.save(filename, self.q_table)

    def load_q_table(self, filename='q_table_sarsa.npy'):
        """
        Loads a Q-table from a file.

        Parameters:
        - filename: The name of the file from which to load the Q-table. The default name is 'q_table.npy'.
        
        Returns:
        - q_table: The loaded Q-table as a NumPy array.
        """
        self.q_table =  np.load(filename)