import torch
from torch import nn
import numpy as np
from collections import deque, namedtuple

class DeepConvQNetwork(nn.module):
    """
    Deep Convolutional Q-Network for reinforcement learning.

    This neural network is designed for use in deep reinforcement learning scenarios,
    particularly with environments that have image-based state representations.

    Args:
        input_dim (tuple): Input dimensions representing the shape of the state space.
        output_dim (int): Number of output nodes representing the Q-values for different actions.

    Attributes:
        input_dim (tuple): Input dimensions of the state space.
        output_dim (int): Number of output nodes representing Q-values.
        fc_input_dim (int): Size of the feature vector produced by the convolutional layers.
        conv (nn.Sequential): Convolutional layers of the neural network.
        network (nn.Sequential): regular layers of the neural network.

    Methods:
        forward(state): Performs a forward pass through the neural network.
        feature_size(): Calculates the size of the feature vector produced by the convolutional layer.
    """
    def __init__(self, input_dim, output_dim):
        super(DeepConvQNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_input_dim = self.feature_size()

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.network = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass through the neural network.

        Args:
            state (torch.Tensor): Input state, typically an image or a tensor representing the environment state.

        Returns:
            torch.Tensor: Output Q-values predicted by the neural network for different actions.
        """
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        qvals = self.network(features)
        
        return qvals
    
    def feature_size(self):
        """
        Calculate the size of the feature vector produced by the convolutional layer.

        This method initializes a tensor filled with zeros and passes it through the convolutional layer,
        then computes the size of the resulting feature vector. The size is crucial for determining
        the input dimension of subsequent layers in a neural network.

        Returns:
            int: Size of the feature vector produced by the convolutional layer.
        """
        return self.conv(torch.autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)
    

class DuelingDeepConvQNetwork(nn.Module):
    """
    Dueling Deep Convolutional Q-Network for reinforcement learning.

    This neural network is designed for use in deep reinforcement learning scenarios,
    particularly with environments that have image-based state representations.
    The dueling architecture decomposes Q-values into state values and advantages.

    Args:
        input_dim (tuple): Input dimensions representing the shape of the state space.
        output_dim (int): Number of output nodes representing the Q-values for different actions.

    Attributes:
        input_dim (tuple): Input dimensions of the state space.
        output_dim (int): Number of output nodes representing Q-values.
        fc_input_dim (int): Size of the feature vector produced by the convolutional layers.
        conv (nn.Sequential): Convolutional layers of the neural network.
        value_stream (nn.Sequential): Stream for estimating state values.
        advantage_stream (nn.Sequential): Stream for estimating advantages.

    Methods:
        forward(state): Performs a forward pass through the dueling deep Q-network.
        feature_size(): Calculates the size of the feature vector produced by the convolutional layer.
    """
    def __init__(self, input_dim, output_dim):
        super(DuelingDeepConvQNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_input_dim = self.feature_size()

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Value Stream.
        self.value_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Advantage Stream.
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

    def forward(self, state):
        """
        Perform a forward pass through the dueling deep Q-network.

        Args:
            state (torch.Tensor): Input state, typically an image or a tensor representing the environment state.

        Returns:
            torch.Tensor: Output Q-values computed using the dueling architecture.
                        The Q-values are decomposed into state values and advantages.
        """
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        
        return qvals
    
    def feature_size(self):
        """
        Calculate the size of the feature vector produced by the convolutional layer.

        This method initializes a tensor filled with zeros and passes it through the convolutional layer,
        then computes the size of the resulting feature vector. The size is crucial for determining
        the input dimension of subsequent layers in a neural network.

        Returns:
            int: Size of the feature vector produced by the convolutional layer.
        """
        return self.conv(torch.autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)




"""namedtuple is used to create a special type of tuple object. Namedtuples
always have a specific name (like a class) and specific fields.
In this case I will create a namedtuple 'Experience',
with fields: state, action, reward,  next_state, done.
Usage: for some given variables s, a, r, s, d you can write for example
exp = Experience(s, a, r, s, d). Then you can access the reward
field by  typing exp.reward"""
Experience = namedtuple('Experience',['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplayBuffer(object):
    """
    Class used to store a buffer containing experiences of the RL agent.

    This buffer stores experiences to facilitate experience replay in reinforcement learning.
    It allows appending new experiences, sampling batches of experiences, and provides the
    current length of the buffer.

    Args:
        maximum_length (int): Maximum number of experiences to be stored in the buffer.

    Attributes:
        buffer (deque): A double-ended queue storing experiences with a maximum length.

    Methods:
        append(experience): Appends a new experience to the buffer.
        __len__(): Returns the current length of the buffer.
        sample_batch(n): Sample a batch of experiences from the replay buffer.

    Raises:
        IndexError: If trying to sample more elements than are available in the buffer.
    """
    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        """Append experience to the buffer"""
        self.buffer.append(experience)

    def __len__(self):
        """overload len operator"""
        return len(self.buffer)

    def sample_batch(self, n):
        """
        Sample a batch of experiences from the replay buffer.

        Args:
            n (int): Number of experiences to sample. 

        Returns:
            Tuple of Lists: A tuple containing five lists, each of size n.
                            The lists represent batches of states, actions, rewards, next states, and done variables.
        Raises:
            IndexError: If trying to sample more elements than are available in the buffer.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        
        # Ensure the batch size is at least 1
        n = max(1, n - 1)

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        # Sample (n-1) indices, excluding the last experience
        indices = np.random.choice(
            len(self.buffer)-1,
            size=n,
            replace=False
        )

        # Always include the latest experience in the batch
        indices = np.append(indices, len(self.buffer) - 1)

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)