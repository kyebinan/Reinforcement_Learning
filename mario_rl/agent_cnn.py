import torch
from torch import nn
import numpy as np
from collections import deque, namedtuple

class AgentCNN(nn.module):
    #TODO : Add double and deuling network 
    def __init__(self, input_shape, n_actions, freeze=False):
        super().__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        # Linear layers
        self.network = nn.Sequential(
            self.conv_layers,
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        if freeze:
            self._freeze()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def foward(self, x):
        """
        Perform forward pass through the neural network.

        Args:
            - x (torch.Tensor): Input tensor to be processed by the network.

        Returns:
            - torch.Tensor: Output tensor after passing through the network.
        """
        return self.network(x)
    
    
    def _get_conv_out(self, shape):
        """
        Calculate the output size of the convolutional layers given the input shape.

        Args:
            - shape (tuple): Input shape as a tuple (channels, height, width).

        Returns:
            - int: Output size after passing through the convolutional layers.
        """
        o = self.conv_layers(torch.zeros(1, *shape))
        # np.prod returns the product of array elements over a given axis
        return int(np.prod(o.size()))
    
    
    def _freeze(self):
        """
        Freeze the parameters of the neural network, preventing further gradient computation during training.

        This method sets the `requires_grad` attribute of all parameters in the network to False.

        Note:
            After calling this method, the parameters will not be updated during training.

        """
        for p in self.network.parameters():
            p.requeries_grad = False



class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent."""
    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
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