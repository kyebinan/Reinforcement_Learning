import numpy as np
import random



class AgentAbstract:
    def __init__(self, action_space_size):
        """
        Initializes the RandomAgent with the available action space.
        :param action_space: A list of possible actions the agent can take.
        """
        self.action_space = action_space_size

    def choose_action(self, state):
        pass

    def update_q_table(self, state, action, reward, next_state):
        pass

    def get_state(self, game):
        pass


class RandomAgent:
    def __init__(self, action_space_size):
        """
        Initializes the RandomAgent with the available action space.
        :param action_space: A list of possible actions the agent can take.
        """
        self.action_space = action_space_size

    def choose_action(self, state):
        """
        Selects a random action from the action space.
        :return: The selected action.
        """
        return random.choice(self.action_space)

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space_size = action_space_size

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.action_space_size)
        else:
            action = np.argmax(self.q_table[state])
        return action

    def update_q_table(self, state, action, reward, next_state):
        self.q_table[state, action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action])

    def get_state(self, game):
        pass