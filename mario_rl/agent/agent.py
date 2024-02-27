import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from mario_rl.agent.utils import ExperienceReplayBuffer, AgentCNN


class Agent():
    def __init__(self):
        pass
        # self.input_dims = 
        # self.n_actions = 
        # self.seed = 
        
        # self.agent_mode=
        # self.network_mode=
        # self.test_mode=
        
        # self.batch_size = 
        # self.n_epochs = 
        # self.update_every = 

        # self.gamma = 
        # self.epsilon = 
        # self.eps_min = 
        # self.eps_max = 
        # self.mem_size = 
        # self.Z =
        # self.tau = 

        # self.memory = 
        # self.update_cntr =
        # self.episode = 1

    # def save_model(self):
    #     """TODO : proofread this function """
    #     # Create models folder
    #     if not os.path.isdir(f'models/{self.agent_name}'):
    #         os.makedirs(f'models/{self.agent_name}')
    #     T.save(self.Q_eval.state_dict(), f'./models/{self.agent_name}/{self.agent_name}_EVAL.pth')
    #     T.save(self.Q_next.state_dict(), f'./models/{self.agent_name}/{self.agent_name}_TARGET.pth')

    # def step(self, state, action, reward, next_state, done):
    #     """TODO : proofread this function """
    #     # Save experience in replay memory
    #     exp = ExperienceReplayBuffer(state, action, reward, next_state, done)
    #     self.memory.append(exp)
        
    #     # Learn every update_cntr time steps.
    #     self.update_cntr = (self.update_cntr + 1) % self.update_every
    #     if len(self.memory) > self.batch_size :
    #         experiences = self.memory.sample_batch(self.batch_size)
    #         self.learn(experiences)

    # def choose_action(self, observation):
    #     """TODO : proofread this function """
    #     if np.random.random() > self.epsilon:
    #         state = observation
    #         self.Q_eval.eval()
    #         with T.no_grad():
    #             Q = self.Q_eval.forward(T.from_numpy(state).to(self.Q_eval.device))
    #         self.Q_eval.train()
    #         action = T.argmax(Q).item()
    
    #     else:
    #         action = np.random.choice(np.arange(self.n_actions))
            
    #     return action
    
    # def epsilon_decay(self):
    #     """TODO : proofread this function """
    #     self.epsilon = max(self.eps_min, self.eps_max - ((self.eps_max-self.eps_min)*(self.episode - 1))/(self.Z-1))
    #     #self.epsilon = max(self.eps_min, self.eps_max*((self.eps_min/self.eps_max)**((self.episode - 1)/(self.Z-1)))) 
    #     self.episode +=1

    # def learn(self, samples):
    #     """TODO : proofread this function """
    #     states, actions, rewards, next_states, dones = samples
    #     # Convert to PyTorch tensors
    #     states = T.tensor(np.array(states), requires_grad=True, dtype=T.float32)
    #     next_states = T.tensor(np.array(next_states), requires_grad=True, dtype=T.float32)
    #     actions = T.tensor(actions, dtype=T.long).unsqueeze(-1) 
    #     rewards = T.tensor(rewards, dtype=T.float32)
    #     dones = T.tensor(dones, dtype=T.float32)
        
    #     if self.agent_mode : #DOUBLE
    #         # Double DQN Approach
    #         self.Q_eval.eval()
    #         with T.no_grad():
    #             # Q_Eval over next states to fetch max action arguement to pass to q_next
    #             q_pred = self.Q_eval.forward(next_states).to(self.Q_eval.device)
    #             max_actions = T.argmax(q_pred, dim=1).long().unsqueeze(1)
    #             # Q_Target over next states from actions will be taken based on q_pred's max_actions
    #             q_next = self.Q_next.forward(next_states).to(self.Q_eval.device)
    #         self.Q_eval.train()
    #         q_target = rewards + self.gamma*q_next.gather(1, max_actions)*(1.0 - dones)
    #     else:
    #         # DQN Approach
    #         q_target_next = self.Q_next.forward(next_states).to(self.Q_eval.device).detach().max(dim=1)[0].unsqueeze(1)
    #         q_target = rewards + (self.gamma* q_target_next * (1 - dones))

    #     # Training
    #     for epoch in range(self.n_epochs):
    #         q_eval = self.Q_eval.forward(states).to(self.Q_eval.device).gather(1, actions)
    #         loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)
    #         self.Q_eval.optimizer.zero_grad()
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(self.Q_eval.parameters(), max_norm=1.0)
    #         self.Q_eval.optimizer.step()

    #     # Replace Target Network
    #     self.replace_target_network()


    # def replace_target_network(self):
    #     """TODO : proofread this function """
    #     if self.update_cntr == 0:
    #         model_state_dict = self.Q_eval.state_dict()
    #         self.Q_next.load_state_dict(model_state_dict)
    #         # Soft Update
    #         # for target_param, local_param in zip(self.Q_next.parameters(), self.Q_eval.parameters()):
    #         #     target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)