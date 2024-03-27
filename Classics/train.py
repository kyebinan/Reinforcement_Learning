import sys
import tqdm
import pygame

from agent.algo import QLearningAgent
from game.snake_game import SnakeGame


ALGO = "Random"
STATE_SPACE_SIZE = 256
ACTION_SPACE_SIZE = 4
LEARNING_RATE = 0.1
GAMMA = 0.9
EPSILON = 0.1
RENDER = True
# Control the speed of the game ()
SIMULATION_SPEED = 100
NUM_EPISODES = 1000


agent = QLearningAgent(  
                state_space_size=STATE_SPACE_SIZE,
                action_space_size=ACTION_SPACE_SIZE,
                alpha=LEARNING_RATE, 
                gamma=GAMMA, 
                epsilon=EPSILON,
                epsilon_decay=0.99, 
                epsilon_min=0.01
            )

game = SnakeGame(30, 20)
clock = pygame.time.Clock()

for EPISODE in range(NUM_EPISODES):
    game.reset()
    state, reward, done = game.get_state(), 0, game.done
    action = agent.choose_action(state)

    while not done:
        next_state, reward, done = game.step(action)
        next_action = agent.choose_action(next_state)
        agent.update_q_table(state, action, reward, next_state)
        state, action = next_state, next_action

        if RENDER:
            game.render()
            clock.tick(SIMULATION_SPEED)  

        if done:
            print("Game Over")
            #game.reset()



# for episode in range(num_episodes):
#     state = env.reset()  # Reset the environment to start a new episode
#     action = agent.choose_action(state)  # Choose an action for the current state

#     while not done:
#         next_state, reward, done, _ = env.step(action)  # Take action
#         next_action = agent.choose_action(next_state)  # Choose next action
        
#         # Update Q-table using SARSA
#         agent.update_q_table(state, action, reward, next_state, next_action)
        
#         state, action = next_state, next_action  # Move to the next state and action
