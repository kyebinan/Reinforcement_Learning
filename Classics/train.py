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



