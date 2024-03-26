import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

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


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = (reward + self.gamma *
                          np.amax(self.model(next_state).cpu().data.numpy()))
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state).cpu().data.numpy()
            target_f[0][action] = target
            target_f = torch.FloatTensor(target_f)
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = nn.MSELoss()(output, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


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
    
class PolicyGradientAgent(AgentAbstract):
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



# agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
# episodes = 1000

# for e in range(episodes):
#     state = env.reset()
#     state = np.reshape(state, [1, state_size])
#     for time in range(500):
#         action = agent.act(state)
#         next_state, reward, done, _ = env.step(action)
#         next_state = np.reshape(next_state, [1, state_size])
#         agent.remember(state, action, reward, next_state, done)
#         state = next_state
#         if done:
#             print("episode: {}/{}, score: {}".format(e, episodes, time))
#             break
#     if len(agent.memory) > 32:
#         agent.replay(32)
