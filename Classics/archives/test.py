import sys
import pygame
import matplotlib.pyplot as plt 
import numpy as np

from agent.dqlearning import DQNAgent
from agent.qlearning import QLearningAgent
from agent.sarsa import SARSAAgent
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

def QLearning():
    agent = QLearningAgent(  
                    state_space_size=STATE_SPACE_SIZE,
                    action_space_size=ACTION_SPACE_SIZE,
                    alpha=LEARNING_RATE, 
                    gamma=GAMMA, 
                    epsilon=EPSILON,
                    epsilon_decay=EPSILON_DECAY, 
                    epsilon_min=EPSILON_MIN
                )

    agent.load_q_table(filename='./models/q_table.npy')

    game = SnakeGame(30, 20)
    clock = pygame.time.Clock()

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



def Sarsa():
    agent = SARSAAgent(  
                state_space_size=STATE_SPACE_SIZE,
                action_space_size=ACTION_SPACE_SIZE,
                alpha=LEARNING_RATE, 
                gamma=GAMMA, 
                epsilon=EPSILON,
                epsilon_decay=EPSILON_DECAY, 
                epsilon_min=EPSILON_MIN
            )

    agent.load_q_table(filename='./models/q_table_sarsa.npy')

    game = SnakeGame(30, 20)
    clock = pygame.time.Clock()

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


def DQLearning():
    STATE_SPACE_SIZE = 8
    BATCH_SIZE = 50
    LEARNING_RATE = 0.001
    agent = DQNAgent(  
                    state_space_size=STATE_SPACE_SIZE,
                    action_space_size=ACTION_SPACE_SIZE,
                    alpha=LEARNING_RATE, 
                    gamma=GAMMA, 
                    epsilon=EPSILON,
                    epsilon_decay=EPSILON_DECAY, 
                    epsilon_min=EPSILON_MIN
                )

    agent.load(filename='./models/dqn_model.pth')

    game = SnakeGame(30, 20)
    clock = pygame.time.Clock()

    state, reward, done = game.get_state_grid(), 0, game.done
    action = agent.choose_action(state)

    total_episode_reward = 0.

    while not done:
        _, reward, done = game.step(action)
        next_state = game.get_state_grid()
        next_action = agent.choose_action(next_state)
        state, action = next_state, next_action

        # Update episode reward
        total_episode_reward += reward

        if RENDER:
            game.render()
            clock.tick(SIMULATION_SPEED)  

        if done:
            print(f"Game Over - Reward : {total_episode_reward}")




# Check if at least one argument is provided
if len(sys.argv) > 1:
    match sys.argv[1]:
        case "qlearning":
            QLearning()
        case "sarsa":
            Sarsa()
        case "dqlearning":
            DQLearning()
    
else:
    print("No arguments provided.")
    print("Run python3 `algorithm`")
    print("`algorithm` can be qlearnig, sarsa, dqlearning")
    print()    
    print("Run Q-Learning by default.")
    QLearning()

        