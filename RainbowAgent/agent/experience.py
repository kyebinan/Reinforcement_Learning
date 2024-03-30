import numpy as np
from collections import deque

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