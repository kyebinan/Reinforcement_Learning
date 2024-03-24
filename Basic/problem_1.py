

# BINAN KOUASSI YOHAN EMMANUEL => 20010402-T456
# SEBASTIAN WENK => 20000825-T431




import numpy as np
import matplotlib.pyplot as plt
import time
import random
from IPython import display


# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED     = '#FFC4CC';
LIGHT_GREEN   = '#95FD99';
BLACK         = '#000000';
WHITE         = '#FFFFFF';
GREY          = '#D1CDDC';
LIGHT_PURPLE  = '#E8D0FF';
LIGHT_ORANGE  = '#FAE0C3';
BRIGHT_PURPLE = '#8000FF';
BRIGHT_ORANGE = '#F57A09';


class Maze:

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
    STEP_REWARD = -5
    GOAL_REWARD = 10
    IMPOSSIBLE_REWARD = -20
    END_REWARD = -75
    KEY_REWARD = 75

    POSITION_KEY = (0,7)


    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze."""
        self.maze                     = maze
        self.actions                  = self.dict_actions()
        self.n_actions                = len(self.actions)
        self.num_states_to_position   = self.dict_states()[0]
        self.position_states_to_num   = self.dict_states()[1]
        self.n_states                 = len(self.num_states_to_position)
        self.transition_probabilities = self.compute_transitions()
        self.rewards                  = self.compute_rewards()


        self.new_num_states_to_position   = self.new_dict_states()[0]
        self.new_position_states_to_num   = self.new_dict_states()[1]
        self.new_n_states                 = len(self.new_num_states_to_position)
        self.transition_new_probabilities = self.compute_new_transitions()
        self.new_rewards                  = self.compute_new_rewards()



    def trade_off_minotaur_move(self, list_move, pos_agent):
    	if random.random() < 0.35 :
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

    	return num_states_to_position, position_states_to_num


    def new_dict_states(self):
    	"""Compute two dictionnairies
    	states : which allow to map a state s with an index (y,x)
    	map    : which allow to map an index (y, x) with a state s"""

    	num_states_to_position = dict()
    	position_states_to_num = dict()
    	s = 0

    	for i in range(self.maze.shape[0]):
    		for j in range(self.maze.shape[1]):
    			for k in range(2):
	    			if self.maze[i,j] != 1:
	    				for ii in range(self.maze.shape[0]):
	    					for jj in range(self.maze.shape[1]):
	    						num_states_to_position[s] = [(i,j), (ii,jj), k]
	    						position_states_to_num[((i,j), (ii,jj),k) ] = s
	    						s += 1

    	return num_states_to_position, position_states_to_num


    def compute_transitions(self):
        """ Computes the transition probabilities for every state action pair.
        	:return numpy.tensor transition probabilities: tensor of transition
        	probabilities of dimension S*S*A
    	"""
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
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

        return transition_probabilities;


    def compute_new_transitions(self):
        """ Computes the transition probabilities for every state action pair.
        	:return numpy.tensor transition probabilities: tensor of transition
        	probabilities of dimension S*S*A
    	"""
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.new_n_states,self.new_n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in self.new_num_states_to_position.keys():
        	for a in range(self.n_actions):
        		init_pos = self.new_num_states_to_position[s][0]
        		int_pos_min = self.new_num_states_to_position[s][1]
        		caught_key = self.new_num_states_to_position[s][2]

        		next_pos = self.agent_move(init_pos, a)
        		next_pos_min_list = self.minotaur_move(int_pos_min) # Be careful it is an list
        		n = len(next_pos_min_list)
        		for next_pos_min in next_pos_min_list:
        			if next_pos == self.POSITION_KEY and caught_key == 0:
	        			pos = (next_pos, next_pos_min, 1)
	        		else:
	        			pos = (next_pos, next_pos_min, caught_key)
	        		next_s = self.new_position_states_to_num[pos]
	        		transition_probabilities[next_s, s, a] = 1/n

        return transition_probabilities;


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
            action = random.choice(possible_moves)

        elif (x == 0 or x == x_max):
            possible_moves = hash_map_x[x]
            action = random.choice(possible_moves)

        elif (y == 0 or y == y_max):
            possible_moves = hash_map_y[y]
            action = random.choice(possible_moves)

        else :
        	possible_moves = [self.MOVE_LEFT, self.MOVE_RIGHT, self.MOVE_UP, self.MOVE_DOWN]
        	action = random.choice(possible_moves)

        # Compute the future position given current (state, action)
        #row = y + self.actions[action][0];
        #col = x + self.actions[action][1];
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
    					rewards[s,a] = self.IMPOSSIBLE_REWARD  #+ self.STEP_REWARD
    					# Reward for reaching the exit
    				elif self.maze[self.num_states_to_position[next_s][0]] == 2:
    					rewards[s,a] = self.GOAL_REWARD;
    					# Reward for taking a step to an empty cell that is not the exit
    				else:
    					rewards[s,a] = self.STEP_REWARD 
    	return rewards;


    def compute_new_rewards(self):
    	rewards = np.zeros((self.new_n_states, self.n_actions));
    	# If the rewards are not described by a weight matrix
    	for s in self.new_num_states_to_position.keys():
    		for a in range(self.n_actions):
    			init_pos = self.new_num_states_to_position[s][0]
    			int_pos_min = self.new_num_states_to_position[s][1]
    			caught_key = self.new_num_states_to_position[s][2]

    			next_pos = self.agent_move(init_pos, a)
    			next_pos_min_list = self.minotaur_move(int_pos_min) # Be careful it is an list
    			#Reward for being eaten by the minotaur
    			for next_pos_min in next_pos_min_list:
    				if next_pos == self.POSITION_KEY and caught_key == 0:
    					pos = (next_pos, next_pos_min, 1)
    				else:
    					pos = (next_pos, next_pos_min, caught_key)

    				next_s = self.new_position_states_to_num[pos]

    				if next_pos == next_pos_min:
    					rewards[s,a] = 2*self.END_REWARD
    					# Rewrd for hitting a wall
    				elif next_pos == init_pos and a != self.STAY:
    					rewards[s,a] = self.IMPOSSIBLE_REWARD  

    				elif next_pos == self.POSITION_KEY :
    					if caught_key == 0 :
    						rewards[s,a] = self.KEY_REWARD
    					else:
    						rewards[s,a] = 1.5*self.STEP_REWARD
    					# Reward for reaching the exit
    				elif self.maze[self.new_num_states_to_position[next_s][0]] == 2:
    					if caught_key == 0 :
    						rewards[s,a] = 1.5*self.STEP_REWARD
    					else:
    						rewards[s,a] = self.GOAL_REWARD
    					# Reward for taking a step to an empty cell that is not the exit
    				else:
    					rewards[s,a] = 1.5*self.STEP_REWARD 
    	return rewards;


    def simulateDynProg(self, start_agent, start_minotaur, V, policy):
        """----ADD DOCSTRING ----"""

        #+----ADD COMMENT ----+
        victory =  False
        path = []
        #V, policy = dynamic_programming(self, horizon, self.rewards, self.transition_probabilities)
        horizon = policy.shape[1]
        t = 0; 
        s =  self.position_states_to_num[(start_agent, start_minotaur)]
        path.append(s)
        while t < horizon-1:
            # Move to next state given the policy and the current state
            init_pos = self.num_states_to_position[s][0]
            int_pos_min = self.num_states_to_position[s][1]

            next_pos = self.agent_move(init_pos, policy[s,t])
            next_pos_min_list = self.minotaur_move(int_pos_min)
            n = len(next_pos_min_list) - 1
            next_pos_min = next_pos_min_list[random.randint(0, n)]

            next_s = self.position_states_to_num[(next_pos, next_pos_min)]
            path.append(next_s)

            t +=1
            s = next_s

            if next_pos == next_pos_min :
            	victory = False
            	#new_path = [self.num_states_to_position[s] for s in path]
            	return path, victory, policy

            if next_pos == (6,5) :
            	victory = True
            	#new_path = [self.num_states_to_position[s] for s in path]
            	return path, victory, policy

        victory = True
        #new_path = [self.num_states_to_position[s] for s in path]
        return path, victory, policy



    def simulateValIter(self, start_agent, start_minotaur, V, policy):
        """----ADD DOCSTRING ----"""

        #+----ADD COMMENT ----+
        victory = False
       	path = []
        # We compute the transition probabilities
        #V, policy = value_iteration(self, gamma, epsilon, self.rewards, self.transition_probabilities)
        s =  self.position_states_to_num[(start_agent, start_minotaur)]
        path.append(s)

        # Initialize current state, next state and time
        t = 1;

        init_pos = self.num_states_to_position[s][0]
        int_pos_min = self.num_states_to_position[s][1]

        next_pos = self.agent_move(init_pos, policy[s])
        next_pos_min_list = self.minotaur_move(int_pos_min)
        n = len(next_pos_min_list) - 1
        next_pos_min = next_pos_min_list[random.randint(0, n)]

        next_s = self.position_states_to_num[(next_pos, next_pos_min)]
        # Loop while state is not the goal state
        while True: 
            # Update state
            s = next_s;
            # Move to next state given the policy and the current state
            next_pos = self.agent_move(next_pos, policy[s]);
            next_pos_min_list = self.minotaur_move(next_pos_min)
            n = len(next_pos_min_list) - 1
            next_pos_min = next_pos_min_list[random.randint(0, n)]
            # Add the position in the maze corresponding to the next state to the path
            next_s = self.position_states_to_num[(next_pos, next_pos_min)]
            path.append(s)
            # Update time and state for next iteration
            t +=1;


            if next_pos == next_pos_min :
            	victory = False
            	#new_path = [self.num_states_to_position[s] for s in path]
            	return path, victory, policy

            if next_pos == (6,5) :
            	victory = True
            	#new_path = [self.num_states_to_position[s] for s in path]
            	return path, victory, policy
        
        victory = True
        #new_path = [self.num_states_to_position[s] for s in path]
        return path, victory, policy


    def simulateValIterPoisonnig(self, start_agent, start_minotaur, V, policy,life_expectancy = 30):
        """----ADD DOCSTRING ----"""

        #+----GEOMETRIC DISTRIBUTION FOR THE LIFETIME OF OUR AGENT----+
        p = 1/life_expectancy # p is the probability of decay induced by poison

        #+----ADD COMMENT ----+

        victory = False
       	path = []
        # We compute the transition probabilities
        #V, policy = value_iteration(self, gamma, epsilon, self.rewards, self.transition_probabilities)
        s =  self.position_states_to_num[(start_agent, start_minotaur)]
        path.append(s)

        # Initialize current state, next state and time
        t = 1;

        init_pos = self.num_states_to_position[s][0]
        int_pos_min = self.num_states_to_position[s][1]

        next_pos = self.agent_move(init_pos, policy[s])
        next_pos_min_list = self.minotaur_move(int_pos_min)
        n = len(next_pos_min_list) - 1
        next_pos_min = next_pos_min_list[random.randint(0, n)]

        next_s = self.position_states_to_num[(next_pos, next_pos_min)]
        # Loop while state is not the goal state
        while True: 
            # Update state
            s = next_s;
            # Move to next state given the policy and the current state
            next_pos = self.agent_move(next_pos, policy[s]);
            next_pos_min_list = self.minotaur_move(next_pos_min)
            n = len(next_pos_min_list) - 1
            next_pos_min = next_pos_min_list[random.randint(0, n)]
            # Add the position in the maze corresponding to the next state to the path
            next_s = self.position_states_to_num[(next_pos, next_pos_min)]
            path.append(s)
            # Update time and state for next iteration
            t +=1;


            if next_pos == next_pos_min :
            	victory = False
            	#new_path = [self.num_states_to_position[s] for s in path]
            	return path, victory, policy, "Minotaur"

            if next_pos == (6,5) :
            	victory = True
            	#new_path = [self.num_states_to_position[s] for s in path]
            	return path, victory, policy, "Exit"

            if p > random.random():
                return path, victory, policy, "Poison"
        
        victory = True
        #new_path = [self.num_states_to_position[s] for s in path]
        return path, victory, policy, "Exit"


    def simulateValIterPoisonnigKey(self, start_agent, start_minotaur, V, policy,life_expectancy = 50):
        """----ADD DOCSTRING ----"""

        #+----GEOMETRIC DISTRIBUTION FOR THE LIFETIME OF OUR AGENT----+
        p = 1/life_expectancy # p is the probability of decay induced by poison

        #+----ADD COMMENT ----+

        victory = False
        caught_key = False
        key = 0
       	path = []
        # We compute the transition probabilities
        #V, policy = value_iteration(self, gamma, epsilon, self.rewards, self.transition_probabilities)
        s =  self.new_position_states_to_num[(start_agent, start_minotaur,key)]
        path.append(s)

        # Initialize current state, next state and time
        t = 1;

        init_pos = self.new_num_states_to_position[s][0]
        int_pos_min = self.new_num_states_to_position[s][1]
        key = self.new_num_states_to_position[s][2]

        next_pos = self.agent_move(init_pos, policy[s])
        next_pos_min_list = self.minotaur_move(int_pos_min)
        n = len(next_pos_min_list) - 1
        next_pos_min = next_pos_min_list[random.randint(0, n)]

        if next_pos == self.POSITION_KEY and key == 0:
        	key = 1

        next_s = self.new_position_states_to_num[(next_pos, next_pos_min, key)]
        # Loop while state is not the goal state
        while True: 
            # Update state
            s = next_s;
            if next_pos == self.POSITION_KEY:
            	caught_key = True
            # Move to next state given the policy and the current state
            next_pos = self.agent_move(next_pos, policy[s]);
            next_pos_min_list = self.minotaur_move(next_pos_min)
            #n = len(next_pos_min_list) - 1
            next_pos_min = self.trade_off_minotaur_move(next_pos_min_list, next_pos)#next_pos_min_list[random.randint(0, n)]
            # Add the position in the maze corresponding to the next state to the path
            if next_pos == self.POSITION_KEY and key == 0:
            	key = 1
            next_s = self.new_position_states_to_num[(next_pos, next_pos_min, key)]
            path.append(s)
            # Update time and state for next iteration
            t +=1;


            if next_pos == next_pos_min :
            	victory = False
            	#new_path = [self.num_states_to_position[s] for s in path]
            	return path, victory, policy, "Minotaur"

            if next_pos == (6,5) and caught_key == True:
            	victory = True
            	#new_path = [self.num_states_to_position[s] for s in path]
            	return path, victory, policy, "Exit with key"

            if next_pos == (6,5) and caught_key == False:
            	victory = False
            	#new_path = [self.num_states_to_position[s] for s in path]
            	return path, victory, policy, "Exit without key"

            if p > random.random():
                return path, victory, policy, "Poison"
        
        victory = True
        #new_path = [self.num_states_to_position[s] for s in path]
        return path, victory, policy, "Exit"





def dynamic_programming(env, horizon, rewards, proba):
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
    p         = proba
    r         = rewards
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


def value_iteration(env, gamma, epsilon, rewards, proba, n_states):
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
    p         = proba
    r         = rewards
    #n_states  = n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

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


def animate_solution(maze, path, policy, env, method):

	def draw_arrow(ax, cell, direction, color='black'):
	    y, x = cell
	    dx, dy = 0, 0
	    if direction == 3:
	        dy = -0.25
	    elif direction == 4:
	        dy = 0.25
	    elif direction == 2:
	        dx = 0.25
	    elif direction == 1:
	        dx = -0.25
	    elif direction == 0:
	        # Option to indicate a stop (for example, a point or a small star)
	        ax.scatter(x+0.25, y-0.25, color=color, marker='*')
	        return
	    ax.arrow(x+0.25, y+0.25, dx, dy, head_width=0.2, head_length=0.2, fc=color, ec=color)

	actions_names = {
        0: "stay",
        1: "left",
        2: "right",
        3: "up",
        4: "down"
    }

	# Map a color to each cell in the maze
	col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: BRIGHT_PURPLE, 4: GREY, 5: BRIGHT_ORANGE, -6: LIGHT_RED, -1: LIGHT_RED};
	# Size of the maze
	rows,cols = maze.shape;

	

	if method == 'ValIter-Key':
		new_path = [env.new_num_states_to_position[s] for s in path]
		path_agent = [pos[0] for pos in new_path]
		path_minotaur = [pos[1] for pos in new_path]
		path_key = [pos[2] for pos in new_path]

	else:
		new_path = [env.num_states_to_position[s] for s in path]
		path_agent = [pos[0] for pos in new_path]
		path_minotaur = [pos[1] for pos in new_path]

	# Create figure of the size of the maze
	fig = plt.figure(1, figsize=(cols,rows))
	# Remove the axis ticks and add title title
	ax = plt.gca()
	ax.set_title('Policy simulation')
	ax.set_xticks([])
	ax.set_yticks([])

	# Give a color to each cell
	colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]
	# Create figure of the size of the maze
	fig = plt.figure(1, figsize=(cols,rows))

	# Create a table to color
	grid = plt.table(cellText=None,cellColours=colored_maze,cellLoc='center',loc=(0,0),edges='closed')

	# Modify the hight and width of the cells in the table
	tc = grid.properties()['children']
	for cell in tc:
		cell.set_height(1.0/rows)
		cell.set_width(1.0/cols)

	# Update the color at each frame
	for i in range(len(path_agent)):
		if method == 'ValIter-Key':
			_, pos, key = env.new_num_states_to_position[path[i]]

		else:
			_, pos = env.num_states_to_position[path[i]]


		for ii in range(maze.shape[0]):
			for jj in range(maze.shape[1]):
				if maze[(ii,jj)] != 1:
					if method == 'ValIter':
						s = env.position_states_to_num[((ii,jj), pos)]
						grid.get_celld()[(ii,jj)].get_text().set_text(actions_names[policy[s]])

					elif method == 'DynProg':
						s = env.position_states_to_num[((ii,jj), pos)]
						grid.get_celld()[(ii,jj)].get_text().set_text(actions_names[policy[s,i]])

					elif method == 'ValIter-Key':
						s = env.new_position_states_to_num[((ii,jj), pos, key)]
						grid.get_celld()[(ii,jj)].get_text().set_text(actions_names[policy[s]])
					#draw_arrow(ax, (jj, ii), policy[s])

		grid.get_celld()[(path_agent[i])].set_facecolor(BRIGHT_ORANGE)
		grid.get_celld()[(path_agent[i])].get_text().set_text('Player')
		# MINOTAUR FRAME 
		grid.get_celld()[(path_minotaur[i])].set_facecolor(BRIGHT_PURPLE)
		grid.get_celld()[(path_minotaur[i])].get_text().set_text('Minotaur')
		# EXIT FRAME
		grid.get_celld()[(6,5)].set_facecolor(LIGHT_GREEN)
		grid.get_celld()[(6,5)].get_text().set_text('Exit')

		display.display(fig)
		display.clear_output(wait=True)
		time.sleep(1)

		grid.get_celld()[(path_agent[i])].set_facecolor(col_map[maze[path_agent[i]]])
		grid.get_celld()[(path_agent[i])].get_text().set_text('')
		grid.get_celld()[(path_minotaur[i])].set_facecolor(col_map[maze[path_minotaur[i]]])
		grid.get_celld()[(path_minotaur[i])].get_text().set_text('')
		
		ax.clear()
		ax.set_title('Policy simulation')
		ax.set_xticks([])
		ax.set_yticks([])

		# Give a color to each cell
		colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]
		# Create a table to color
		grid = plt.table(cellText=None,cellColours=colored_maze,cellLoc='center',loc=(0,0),edges='closed')

		# Modify the hight and width of the cells in the table
		tc = grid.properties()['children']
		for cell in tc:
			cell.set_height(1.0/rows)
			cell.set_width(1.0/cols)


def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);





































