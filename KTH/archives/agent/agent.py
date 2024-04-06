from algo import RandomAgent, QLearningAgent


class Agent:
    def __init__(self, state_space_size, action_space_size, algorithm="Random", alpha=0.1, gamma=0.9, epsilon=0.1):
        self.algorithm = algorithm
        
        match self.algorithm:
            case "Random":
                self.agent = RandomAgent()
            case "QLearning":
                self.agent = None
            case "SARSA":
                self.agent = None
            case "DQLearning":
                self.agent = None
            case "PolicyGradient":
                self.agent = None
            case "ActorCritic":
                self.agent = None

    def choose_action(self, state):
        return self.agent.choose_action(state)

    def update_q_table(self, state, action, reward, next_state):
        self.agent.update_q_table(self, state, action, reward, next_state)

    def get_state(self, game):
        """
        This method would need specific implementation based on the game's state representation.
        """
        pass