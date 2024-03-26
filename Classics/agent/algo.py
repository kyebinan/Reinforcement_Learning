import numpy as np
import random

class AgentAbstract:
    def __init__(self, action_space_size):
        self.action_space_size = action_space_size

    def choose_action(self, state):
        """
        To be implemented by subclasses.
        """
        raise NotImplementedError

    def update_q_table(self, state, action, reward, next_state):
        """
        To be implemented by subclasses.
        """
        raise NotImplementedError

    def get_state(self, game):
        """
        To be implemented by subclasses.
        """
        raise NotImplementedError

class RandomAgent(AgentAbstract):
    def __init__(self, action_space_size):
        super().__init__(action_space_size)

    def choose_action(self, state):
        """
        Selects a random action from the action space.
        :return: The selected action.
        """
        return random.randint(0, self.action_space_size - 1)

    def update_q_table(self, state, action, reward, next_state):
        """
        Not applicable for RandomAgent but implemented to fulfill the interface.
        """
        pass

    def get_state(self, game):
        """
        Not applicable for RandomAgent but implemented to fulfill the interface.
        """
        pass

class QLearningAgent(AgentAbstract):
    def __init__(self, state_space_size, action_space_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(action_space_size)
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, self.action_space_size - 1)
        else:
            action = np.argmax(self.q_table[state])
        return action

    def update_q_table(self, state, action, reward, next_state):
        self.q_table[state, action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action])

    def get_state(self, game):
        """
        This method would need specific implementation based on the game's state representation.
        """
        pass

class SARSAAgent(AgentAbstract):
    def __init__(self, state_space_size, action_space_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(action_space_size)
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

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

    def get_state(self, game):
        """
        This method would need specific implementation based on the game's state representation.
        """
        pass
