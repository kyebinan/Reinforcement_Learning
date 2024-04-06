class Game:
    def __init__(self):
        self.score = 0
        self.done = False

    def reset(self):
        """
        Reset the game to its initial state.
        """
        self.score = 0
        self.done = False
        # Initialize game state

    def step(self, action):
        """
        Apply an action to the game, return the new state, reward, and done status.
        :param action: The action to take.
        :return: A tuple of (state, reward, done).
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def render(self):
        """
        Render the game state to the screen. Useful for debugging.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_state(self):
        """
        Return the current state of the game.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass
