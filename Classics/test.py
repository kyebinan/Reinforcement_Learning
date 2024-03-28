import sys
import numpy as np
import matplotlib.pyplot as plt
import pygame


from agent.qlearning import QLearningAgent
from game.snake_game import SnakeGame

STATE_SPACE_SIZE = 256
ACTION_SPACE_SIZE = 4
LEARNING_RATE = 0.1
GAMMA = 0.9
EPSILON = 0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01 
RENDER = True
SIMULATION_SPEED = 10
NUM_EPISODES = 2000
N_EP_RUNNING_AVG = 50


agent = QLearningAgent(  
                state_space_size=STATE_SPACE_SIZE,
                action_space_size=ACTION_SPACE_SIZE,
                alpha=LEARNING_RATE, 
                gamma=GAMMA, 
                epsilon=EPSILON,
                epsilon_decay=EPSILON_DECAY, 
                epsilon_min=EPSILON_MIN
            )

agent.load_q_table()

game = SnakeGame(30, 20)
clock = pygame.time.Clock()
episode_reward_list = []

state, reward, done = game.get_state(), 0, game.done
action = agent.choose_action(state)

total_episode_reward = 0.

while not done:
    next_state, reward, done = game.step(action)
    next_action = agent.choose_action(next_state)
    state, action = next_state, next_action

    # Update episode reward
    total_episode_reward += reward

    if RENDER:
        game.render()
        clock.tick(SIMULATION_SPEED)  

    if done:
        print(f"Game Over - Reward : {total_episode_reward}")