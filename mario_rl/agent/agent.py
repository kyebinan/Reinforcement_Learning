import random
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.optim as optim
import os

from agent.network import DeepConvQNetwork, DuelingDeepConvQNetwork
from agent.experience import ExperienceReplayBuffer
from agent.utils import Experience

# NN Architectures
SIMPLE="SIMPLE"
DOUBLE="DOUBLE"
DUELING="DUELING"
# Training or exploitation 
TRAIN="TRAIN"
TEST="TEST"

class Agent():
    def __init__(self, 
                 input_dims, 
                 n_actions,
                 max_mem_size, 
                 Z,
                 agent_mode=SIMPLE, 
                 network_mode=SIMPLE, 
                 batch_size=64, 
                 n_epochs=5, 
                 lr=0.001, 
                 gamma=0.95, 
                 eps_min=0.05, 
                 eps_max=0.95):
        
        self.input_dims = input_dims
        self.n_actions =  n_actions
        self.seed = random.seed(42)
        
        self.agent_mode=agent_mode
        self.network_mode=network_mode 
        
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.update_every = max_mem_size//batch_size # No real explanation, just a good practice
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_max
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.mem_size = max_mem_size
        self.Z = Z

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
        elif network_mode==SIMPLE:
            self.Q_eval = DeepConvQNetwork(input_dim=self.input_dims, output_dim=self.n_actions)
            self.Q_next = DeepConvQNetwork(input_dim=self.input_dims, output_dim=self.n_actions)
        else:
            raise ValueError("the network_mode must be SIMPLE or DUELING.")

        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=lr)


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
        exp = Experience(state, action, reward, next_state, done)
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
            observation = T.tensor(np.array(observation), dtype=T.float32).unsqueeze(0).to(self.Q_eval.device)
            action = self.Q_eval(observation).argmax().item()
            # observation = np.array(observation, dtype=np.uint8)
            # self.Q_eval.eval()
            # with T.no_grad():
            #     Q = self.Q_eval.forward(T.from_numpy(observation).to(self.Q_eval.device))
            # self.Q_eval.train()
            # action = T.argmax(Q).item()
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
        
        if self.agent_mode == DOUBLE:
            # Double DQN Approach
            self.Q_eval.eval()
            with T.no_grad():
                q_pred = self.Q_eval.forward(next_states).to(self.Q_eval.device)
                max_actions = T.argmax(q_pred, dim=1).long().unsqueeze(1)
                q_next = self.Q_next.forward(next_states).to(self.Q_eval.device)
            self.Q_eval.train()
            q_target = rewards.to(self.Q_eval.device) + self.gamma * q_next.gather(1, max_actions).to(self.Q_eval.device) * (1.0 - dones.to(self.Q_eval.device))

        elif self.agent_mode == SIMPLE:
            # DQN Approach
            q_target_next = self.Q_next.forward(next_states).to(self.Q_eval.device).detach().max(dim=1)[0].unsqueeze(1)
            q_target = rewards.to(self.Q_eval.device) + (self.gamma * q_target_next * (1 - dones.to(self.Q_eval.device)))

        else:
            raise ValueError("the agent_mode must be SIMPLE or DOUBLE.")

        # Training
        for _ in range(self.n_epochs):
            q_eval = self.Q_eval.forward(states)
            actions = actions.to(self.Q_eval.device)  
            q_eval = q_eval.gather(1, actions)
            q_target = q_target.to(self.Q_eval.device)  
            loss = F.mse_loss(q_eval, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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