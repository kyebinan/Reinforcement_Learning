import numpy as np
import random
from abc import ABC, abstractmethod

class Algorithm:

    @staticmethod
    def dynamic_programming(env, horizon):
        """ Solves the shortest path problem using dynamic programming
            :input Maze env           : The maze environment in which we seek to
                                        find the shortest path.
            :input int horizon        : The time T up to which we solve the problem.
            :return numpy.array V     : Optimal values for every state at every
                                        time, dimension S*T
            :return numpy.array policy: Optimal time-varying policy at every state,
                                        dimension S*T
        """

        # The dynamic prgramming requires the knowledge of :
        # - Transition probabilities
        # - Rewards
        # - State space
        # - Action space
        # - The finite horizon
        p         = env.transition_probabilities
        r         = env.rewards
        n_states  = env.n_states;
        n_actions = env.n_actions;
        T         = horizon;

        # The variables involved in the dynamic programming backwards recursions
        V      = np.zeros((n_states, T+1));
        policy = np.zeros((n_states, T+1));
        Q      = np.zeros((n_states, n_actions));

        # Initialization
        Q            = np.copy(r);
        V[:, T]      = np.max(Q,1);
        policy[:, T] = np.argmax(Q,1);

        # The dynamic programming bakwards recursion
        for t in range(T-1,-1,-1):
            # Update the value function acccording to the bellman equation
            for s in range(n_states):
                for a in range(n_actions):
                    # Update of the temporary Q values
                    Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
            # Update by taking the maximum Q value w.r.t the action a
            V[:,t] = np.max(Q,1);
            # The optimal action is the one that maximizes the Q function
            policy[:,t] = np.argmax(Q,1);
        return V, policy;


    @staticmethod
    def value_iteration(env, gamma, epsilon):
        """ Solves the shortest path problem using value iteration
            :input Maze env           : The maze environment in which we seek to
                                        find the shortest path.
            :input float gamma        : The discount factor.
            :input float epsilon      : accuracy of the value iteration procedure.
            :return numpy.array V     : Optimal values for every state at every
                                        time, dimension S*T
            :return numpy.array policy: Optimal time-varying policy at every state,
                                        dimension S*T
        """
        # The value itearation algorithm requires the knowledge of :
        # - Transition probabilities
        # - Rewards
        # - State space
        # - Action space
        # - The finite horizon
        p         = env.transition_probabilities
        r         = env.rewards
        n_states  = env.n_states
        n_actions = env.n_actions

        # Required variables and temporary ones for the VI to run
        V   = np.zeros(n_states);
        Q   = np.zeros((n_states, n_actions));
        BV  = np.zeros(n_states);
        # Iteration counter
        n   = 0;
        # Tolerance error
        tol = (1 - gamma)* epsilon/gamma

        # Initialization of the VI
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);

        # Iterate until convergence
        while np.linalg.norm(V - BV) >= tol and n < 200:
            # Increment by one the numbers of iteration
            n += 1;
            # Update the value function
            V = np.copy(BV);
            # Compute the new BV
            for s in range(n_states):
                for a in range(n_actions):
                    Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
            BV = np.max(Q, 1);
            # Show error
            #print(np.linalg.norm(V - BV))

        # Compute policy
        policy = np.argmax(Q,1);
        # Return the obtained policy
        return V, policy;



class Maze(ABC):
    
    def dict_actions(self):
        """Compute the delta for each action and save these in a dictionnary"""
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    def agent_move(self, init_pos, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        y,x = init_pos
        row = y + self.actions[action][0];
        col = x + self.actions[action][1];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1);
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return init_pos
        else:
            return (row, col)
        
    def minotaur_move(self, init_pos):
        pass
    
    @abstractmethod
    def dict_states(self):
        pass

    @abstractmethod
    def compute_transitions(self):
        pass

    @abstractmethod
    def compute_rewards(self):
        pass
    

class SimpleMaze(Maze):

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 10
    IMPOSSIBLE_REWARD = -10
    

    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze."""
        self.maze                     = maze
        self.actions                  = self.dict_actions()
        self.n_actions                = len(self.actions)
        self.num_states_to_position, self.position_states_to_num   = self.dict_states()
        self.n_states                 = len(self.num_states_to_position)
        self.transition_probabilities = self.compute_transitions()
        self.rewards                  = self.compute_rewards()
        self.exit = (6,5)
    

    def dict_states(self):
        """Compute two dictionnairies
        states : which allow to map a state s with an index (y,x)
        map    : which allow to map an index (y, x) with a state s"""

        num_states_to_position = dict()
        position_states_to_num = dict()
        s = 0

        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i,j] != 1:
                    num_states_to_position[s] = (i,j)
                    position_states_to_num[(i,j)] = s
                    s += 1
                if self.maze[i,j] == 2 :
                    self.exit = (i,j)

        return num_states_to_position, position_states_to_num
    

    def compute_transitions(self):
        """ Computes the transition probabilities for every state action pair.
        :return numpy.tensor transition probabilities: tensor of transition
        probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. 
        # Note that the transitions are deterministic.
        for s in self.num_states_to_position.keys():
            for a in range(self.n_actions):
                next_pos = self.agent_move(self.num_states_to_position[s], a)
                next_s = self.position_states_to_num[next_pos]
                transition_probabilities[next_s, s, a] = 1.
        return transition_probabilities;


    def compute_rewards(self):
        rewards = np.zeros((self.n_states, self.n_actions));
        # If the rewards are not described by a weight matrix
        for s in self.num_states_to_position.keys():
            for a in range(self.n_actions):
                next_pos = self.agent_move(self.num_states_to_position[s], a)
                pos = next_pos
                next_s = self.position_states_to_num[pos]
                if next_pos == self.num_states_to_position[s] and a != self.STAY:
                    rewards[s,a] = self.IMPOSSIBLE_REWARD
                    # Reward for reaching the exit
                elif self.maze[self.num_states_to_position[next_s]] == 2:
                    rewards[s,a] = self.GOAL_REWARD;
                    # Reward for taking a step to an empty cell that is not the exit
                else:
                    rewards[s,a] = self.STEP_REWARD 
        return rewards;


    def simulateDynProg(self, start_agent, V, policy):
        victory =  False
        path = []
        horizon = policy.shape[1]
        t = 0; 
        s =  self.position_states_to_num[start_agent]
        path.append(s)
        while t < horizon-1:
            # Move to next state given the policy and the current state
            init_pos = self.num_states_to_position[s]
            next_pos = self.agent_move(init_pos, policy[s,t])
            s = self.position_states_to_num[next_pos]
            path.append(s)
            t +=1
            
            if next_pos == self.exit :
                victory = True
                return path, victory, policy

        return path, victory, policy


class MinotaurMaze(Maze):

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -10
    GOAL_REWARD = 100
    IMPOSSIBLE_REWARD = -20
    END_REWARD = -50


    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze."""
        self.maze                     = maze
        self.actions                  = self.dict_actions()
        self.n_actions                = len(self.actions)
        self.num_states_to_position, self.position_states_to_num   = self.dict_states()
        self.n_states                 = len(self.num_states_to_position)
        self.transition_probabilities = self.compute_transitions()
        self.rewards                  = self.compute_rewards()
        self.exit = (6,5)


    def trade_off_minotaur_move(self, list_move, pos_agent):
        if random.random() < 0.5 :
            min_dist = abs(pos_agent[1]-list_move[0][1]) + abs(pos_agent[0]-list_move[0][0])
            dist, i = 0, 0
            for pos in list_move[1:]:
                i += 1
                dist = abs(pos_agent[1]-pos[1]) + abs(pos_agent[0]-pos[0])
                if dist < min_dist :
                    min_dist = dist 
            return list_move[i]

        else:
            n = len(list_move) - 1
            return list_move[random.randint(0, n)]


    def dict_actions(self):
        """Compute the delta for each action and save these in a dictionnary"""
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;


    def dict_states(self):
        """Compute two dictionnairies
        states : which allow to map a state s with an index (y,x)
        map    : which allow to map an index (y, x) with a state s"""

        num_states_to_position = dict()
        position_states_to_num = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i,j] != 1:
                    for ii in range(self.maze.shape[0]):
                        for jj in range(self.maze.shape[1]):
                            num_states_to_position[s] = [(i,j), (ii,jj)]
                            position_states_to_num[((i,j), (ii,jj))] = s
                            s += 1
                if self.maze[i,j] == 2 :
                    self.exit = (i,j)

        return num_states_to_position, position_states_to_num
    

    def compute_transitions(self):
        """ Computes the transition probabilities for every state action pair.
        :return numpy.tensor transition probabilities: tensor of transition
        probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. 
        # Note that the transitions are deterministic.
        for s in self.num_states_to_position.keys():
            for a in range(self.n_actions):
                init_pos = self.num_states_to_position[s][0]
                int_pos_min = self.num_states_to_position[s][1]
                next_pos = self.agent_move(init_pos, a)
                next_pos_min_list = self.minotaur_move(int_pos_min) # Be careful it is an list
                n = len(next_pos_min_list)
                for next_pos_min in next_pos_min_list:
                    pos = (next_pos, next_pos_min)
                    next_s = self.position_states_to_num[pos]
                    transition_probabilities[next_s, s, a] = 1/n

        return transition_probabilities
        
    
    def minotaur_move(self, init_pos):
        """ Makes a step in the maze, given a current position and an action.
            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # specifies the possible movements according to the border on which the minotaur is located
        y_max, x_max = self.maze.shape[0] - 1, self.maze.shape[1] - 1
        hash_map_y = {0     : (self.MOVE_LEFT, self.MOVE_RIGHT, self.MOVE_DOWN),
                      y_max : (self.MOVE_LEFT, self.MOVE_RIGHT, self.MOVE_UP)}

        hash_map_x = {0     : (self.MOVE_RIGHT, self.MOVE_DOWN, self.MOVE_UP),
                      x_max : (self.MOVE_LEFT,self.MOVE_DOWN, self.MOVE_UP)}

        y,x = init_pos
        possible_moves = []
        if (x == 0 or x == x_max) and (y == 0 or y == y_max) :
            #possible_moves = tuple(set(moves_dico_x[x]) & set(moves_dico_y[y]))
            possible_moves = tuple(item for item in hash_map_x[x] if item in hash_map_y[y])
           
        elif (x == 0 or x == x_max):
            possible_moves = hash_map_x[x]

        elif (y == 0 or y == y_max):
            possible_moves = hash_map_y[y]
            
        else :
            possible_moves = [self.MOVE_LEFT, self.MOVE_RIGHT, self.MOVE_UP, self.MOVE_DOWN]
            

        # Compute the future position given current (state, action)
        new_positions = []
        for act in possible_moves:
            row = y + self.actions[act][0];
            col = x + self.actions[act][1];
            new_positions += [(row, col)]

        
        return new_positions


    def compute_rewards(self):
        rewards = np.zeros((self.n_states, self.n_actions));
        # If the rewards are not described by a weight matrix
        for s in self.num_states_to_position.keys():
            for a in range(self.n_actions):
                init_pos = self.num_states_to_position[s][0]
                int_pos_min = self.num_states_to_position[s][1]

                next_pos = self.agent_move(init_pos, a)
                next_pos_min_list = self.minotaur_move(int_pos_min) # Be careful it is an list
                #Reward for being eaten by the minotaur
                for next_pos_min in next_pos_min_list:
                    pos = (next_pos, next_pos_min)
                    next_s = self.position_states_to_num[pos]
                    if next_pos == next_pos_min:
                        rewards[s,a] = self.END_REWARD
                        # Rewrd for hitting a wall
                    elif next_pos == init_pos and a != self.STAY:
                        rewards[s,a] = self.IMPOSSIBLE_REWARD  
                        # Reward for reaching the exit
                    elif self.maze[self.num_states_to_position[next_s][0]] == 2:
                        rewards[s,a] = self.GOAL_REWARD;
                        # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s,a] = self.STEP_REWARD 
        return rewards;


    def simulateValIter(self, start_agent, start_minotaur, V, policy):
        victory = False
        path = []
        # We compute the transition probabilities
        s =  self.position_states_to_num[(start_agent, start_minotaur)]
        path.append(s)
        # Initialize current state, next state and time
        init_pos = self.num_states_to_position[s][0]
        int_pos_min = self.num_states_to_position[s][1]
        next_pos = self.agent_move(init_pos, policy[s])
        next_pos_min_list = self.minotaur_move(int_pos_min)
        n = len(next_pos_min_list) - 1
        next_pos_min = self.trade_off_minotaur_move(next_pos_min_list, next_pos)
        next_s = self.position_states_to_num[(next_pos, next_pos_min)]
        # Loop while state is not the goal state
        while True: 
            # Update state
            s = next_s;
            # Move to next state given the policy and the current state
            next_pos = self.agent_move(next_pos, policy[s]);
            next_pos_min_list = self.minotaur_move(next_pos_min)
            n = len(next_pos_min_list) - 1
            next_pos_min = next_pos_min = self.trade_off_minotaur_move(next_pos_min_list, next_pos)
            # Add the position in the maze corresponding to the next state to the path
            next_s = self.position_states_to_num[(next_pos, next_pos_min)]
            path.append(s)

            if next_pos == next_pos_min :
                victory = False
                return path, victory, policy

            if next_pos == self.exit :
                victory = True
                return path, victory, policy
        



