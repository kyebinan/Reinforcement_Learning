import sys
import tqdm
import random
import pygame


from agent.agent import Agent
from game.snake_game import SnakeGame
from game.tetris_game import TetrisGame


ALGO = "Random"
STATE_SPACE_SIZE = 600
ACTION_SPACE_SIZE = 4
LEARNING_RATE = 0.1
GAMMA = 0.9
EPSILON = 0.1
RENDER = True
# Control the speed of the game ()
SIMULATION_SPEED = 10  


agent = Agent(  state_space_size=STATE_SPACE_SIZE,
                action_space_size=ACTION_SPACE_SIZE,
                algorithm="Random",
                alpha=LEARNING_RATE, 
                gamma=GAMMA, 
                epsilon=EPSILON
            )

game = SnakeGame(30, 20)
clock = pygame.time.Clock()
grid, reward, _ = game.get_state(), 0, game.done
while not game.done:
    action = agent.choose_action(grid)
    grid, reward, done = game.step(action)

    if RENDER:
        game.render()
        clock.tick(SIMULATION_SPEED)  

    if game.done:
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
