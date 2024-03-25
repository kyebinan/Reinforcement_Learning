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