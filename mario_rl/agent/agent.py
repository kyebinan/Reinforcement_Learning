import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import os
from utils import DeepConvQNetwork, DuelingDeepConvQNetwork, ExperienceReplayBuffer

# NN Architectures
SIMPLE="SIMPLE"
DOUBLE="DOUBLE"
DUELING="DUELING"
# Training or exploitation 
TRAIN="TRAIN"
TEST="TEST"

class Agent():
    def __init__(self, input_dims, n_actions, seed, Z, max_mem_size, agent_mode=SIMPLE, network_mode=SIMPLE, 
                test_mode=False, batch_size=64, n_epochs=1, update_every=5, lr=0.0005, fc1_dims=64, fc2_dims=64,
                gamma=0.99, eps_min=0.05,eps_max=0.95, tau=1e-3):
        
        self.input_dims = input_dims
        self.n_actions =  n_actions
        self.seed = random.seed(seed)
        
        self.agent_mode=agent_mode
        self.network_mode=network_mode
        self.test_mode=test_mode
        
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.update_every = update_every
        
        self.gamma = gamma
        self.epsilon = eps_max
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.mem_size = max_mem_size
        self.Z = Z
        self.tau = tau

        self.memory = ExperienceReplayBuffer(max_mem_size)
        self.update_cntr = 0
        self.episode = 1

        # For naming purpose
        agent_ = '{}-'.format(self.agent_mode) if self.agent_mode!=SIMPLE else ''
        network_ = '{}-'.format(self.network_mode) if self.network_mode!=SIMPLE else ''
        self.agent_name = f'{agent_}{network_}DQN'.strip()

        if network_mode==DUELING:
            self.Q_eval = DuelingDeepConvQNetwork(input_dim=self.input_dims, output_dim=self.n_actions)
            self.Q_next = DuelingDeepConvQNetwork(input_dim=self.input_dims, output_dim=self.n_actions)
        else:
            self.Q_eval = DeepConvQNetwork(input_dim=self.input_dims, output_dim=self.n_actions)
            self.Q_next = DeepConvQNetwork(input_dim=self.input_dims, output_dim=self.n_actions)


    def save_model(self):
        """
        Save the current state of the Q-value estimation models.

        This method creates a 'models' folder, if it does not already exist, and saves the state
        dictionaries of the evaluation and target Q-value estimation models to separate files.

        Saved files follow the naming convention: '<agent_name>_EVAL.pth' for the evaluation model,
        and '<agent_name>_TARGET.pth' for the target model.

        Returns:
            None
        """
        if not os.path.isdir(f'models/{self.agent_name}'):
            os.makedirs(f'models/{self.agent_name}')
        T.save(self.Q_eval.state_dict(), f'./models/{self.agent_name}/{self.agent_name}_EVAL.pth')
        T.save(self.Q_next.state_dict(), f'./models/{self.agent_name}/{self.agent_name}_TARGET.pth')


    def step(self, state, action, reward, next_state, done):
        """
        Process a single step of the reinforcement learning agent.

        Save the current experience in the replay memory and trigger the learning process
        if the update counter reaches the specified update interval.

        Args:
            state (torch.Tensor): Current state of the environment.
            action (int): Chosen action in the current state.
            reward (float): Received reward after taking the chosen action.
            next_state (torch.Tensor): Next state of the environment after taking the action.
            done (bool): Flag indicating whether the episode is terminated after this step.

        Returns:
            None
        """
        # Save experience in replay memory
        exp = ExperienceReplayBuffer(state, action, reward, next_state, done)
        self.memory.append(exp)
        
        # Learn every update_cntr time steps.
        self.update_cntr = (self.update_cntr + 1) % self.update_every
        if len(self.memory) > self.batch_size :
            experiences = self.memory.sample_batch(self.batch_size)
            self.learn(experiences)


    def choose_action(self, observation):
        """
        Select an action using an epsilon-greedy strategy.

        With probability epsilon, a random action is chosen; otherwise, the action
        with the highest Q-value, as predicted by the evaluation model, is selected.

        Args:
            observation (np.ndarray): Current observation or state of the environment.

        Returns:
            int: Chosen action based on the epsilon-greedy strategy.
        """
        if np.random.random() > self.epsilon:
            state = observation
            self.Q_eval.eval()
            with T.no_grad():
                Q = self.Q_eval.forward(T.from_numpy(state).to(self.Q_eval.device))
            self.Q_eval.train()
            action = T.argmax(Q).item()
        else:
            action = np.random.choice(np.arange(self.n_actions))
            
        return action
    
    
    def epsilon_decay(self):
        """
        Update the exploration-exploitation trade-off parameter (epsilon) using a decay strategy.

        The epsilon value is decayed over episodes to gradually shift from exploration to exploitation.

        The decay formula is defined as:
            epsilon = max(eps_min, eps_max - ((eps_max - eps_min) * (episode - 1)) / (Z - 1))

        where:
            eps_min (float): Minimum exploration rate.
            eps_max (float): Maximum exploration rate.
            episode (int): Current episode number.
            Z (int): Total number of episodes for decay.

        Returns:
            None
        """
        self.epsilon = max(self.eps_min, self.eps_max - ((self.eps_max-self.eps_min)*(self.episode - 1))/(self.Z-1)) 
        self.episode +=1


    def learn(self, samples):
        """
        Update the Q-value estimation model based on a batch of experiences.

        Args:
            samples (Tuple of Lists): A tuple containing five lists of experiences, including states,
                                    actions, rewards, next states, and done flags.

        Returns:
            None
        """
        states, actions, rewards, next_states, dones = samples
        # Convert to PyTorch tensors
        states = T.tensor(np.array(states), requires_grad=True, dtype=T.float32)
        next_states = T.tensor(np.array(next_states), requires_grad=True, dtype=T.float32)
        actions = T.tensor(actions, dtype=T.long).unsqueeze(-1) 
        rewards = T.tensor(rewards, dtype=T.float32)
        dones = T.tensor(dones, dtype=T.float32)
        
        if self.agent_mode :
            # Double DQN Approach
            self.Q_eval.eval()
            with T.no_grad():
                # Q_Eval over next states to fetch max action arguement to pass to q_next
                q_pred = self.Q_eval.forward(next_states).to(self.Q_eval.device)
                max_actions = T.argmax(q_pred, dim=1).long().unsqueeze(1)
                # Q_Target over next states from actions will be taken based on q_pred's max_actions
                q_next = self.Q_next.forward(next_states).to(self.Q_eval.device)
            self.Q_eval.train()
            q_target = rewards + self.gamma*q_next.gather(1, max_actions)*(1.0 - dones)

        else:
            # DQN Approach
            q_target_next = self.Q_next.forward(next_states).to(self.Q_eval.device).detach().max(dim=1)[0].unsqueeze(1)
            q_target = rewards + (self.gamma* q_target_next * (1 - dones))

        # Training
        for epoch in range(self.n_epochs):
            q_eval = self.Q_eval.forward(states).to(self.Q_eval.device).gather(1, actions)
            loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)
            self.Q_eval.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.Q_eval.parameters(), max_norm=1.0)
            self.Q_eval.optimizer.step()

        # Replace Target Network
        self.replace_target_network()


    def replace_target_network(self):
        """
        Replace the parameters of the target Q-value estimation model with those of the evaluation model.

        This operation is performed periodically based on the specified update interval.

        Returns:
            None
        """
        if self.update_cntr == 0:
            model_state_dict = self.Q_eval.state_dict()
            self.Q_next.load_state_dict(model_state_dict)