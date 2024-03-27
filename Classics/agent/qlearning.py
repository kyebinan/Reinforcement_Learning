import numpy as np 
import random

class QLearningAgent():
    def __init__(self, state_space_size, action_space_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate of epsilon over time
        self.epsilon_min = epsilon_min  # Minimum value of epsilon

    def choose_action(self, state):
        """
        Selects and returns an action for a given state using an epsilon-greedy strategy.
        
        This method decides whether to take a random action or to choose the best-known action
        based on the current Q-table and the epsilon value. With a probability of `epsilon`, a 
        random action is chosen (exploration), and with a probability of `1 - epsilon`, the 
        action with the highest Q-value for the current state is chosen (exploitation).
        
        Parameters:
        - state: The current state of the environment as an integer. The state should be an index
                that directly corresponds to a row in the Q-table, where each column represents 
                a possible action.
        
        Returns:
        - action: An integer representing the chosen action. The action is an index that 
                corresponds to the column in the Q-table for the given state.
        """
        if np.random.rand() < self.epsilon:
            action = random.randint(0, self.action_space_size - 1)
        else:
            action = np.argmax(self.q_table[state])
        return action

    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q-value for a given state-action pair based on the observed
        reward and the maximum Q-value of the next state. This method implements
        the Q-learning update rule.

        Parameters:
        - state: The current state of the environment, represented as an integer.
                This should be an index that directly corresponds to a row in
                the Q-table.
        - action: The action taken in the current state, represented as an integer.
                This should be an index that directly corresponds to a column
                in the Q-table for the given state.
        - reward: The immediate reward received after taking the action in the
                current state.
        - next_state: The state of the environment after taking the action, also
                    represented as an integer. Like the current state, this should
                    directly correspond to a row in the Q-table.
        Returns: None. 
        Note:
        The Q-learning formula used for the update is as follows:
            Q(state, action) = Q(state, action) + alpha * (td_target - Q(state, action))
        where: td_target = reward + gamma * max(Q(next_state, all actions))
        """
        # Q-learning formula
        best_next_action = np.argmax(self.q_table[next_state])  # Best action from the next state
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_delta

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