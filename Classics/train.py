import sys
import numpy as np
import matplotlib.pyplot as plt
import pygame

from agent.qlearning import QLearningAgent
from agent.sarsa import SARSAAgent
from agent.dqlearning import DQNAgent

from game.snake_game import SnakeGame


STATE_SPACE_SIZE = 256
ACTION_SPACE_SIZE = 4
LEARNING_RATE = 0.1
GAMMA = 0.9
EPSILON = 1
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01 
RENDER = False
SIMULATION_SPEED = 100
NUM_EPISODES = 2000
N_EP_RUNNING_AVG = 50



def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


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

    game = SnakeGame(30, 20)
    clock = pygame.time.Clock()
    episode_reward_list = []

    for EPISODE in range(NUM_EPISODES):
        game.reset()
        state, reward, done = game.get_state(), 0, game.done
        action = agent.choose_action(state)
        total_episode_reward = 0.

        while not done:
            next_state, reward, done = game.step(action)
            next_action = agent.choose_action(next_state)
            agent.update_q_table(state, action, reward, next_state)
            state, action = next_state, next_action

            # Update episode reward
            total_episode_reward += reward

            if RENDER:
                game.render()
                clock.tick(SIMULATION_SPEED)  

            if done:
                print(f"Game Over - Episode {EPISODE} - Reward : {total_episode_reward}")

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
        agent.decay_epsilon()
                
    avg_reward = np.mean(episode_reward_list)
    confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(NUM_EPISODES)


    print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
                    avg_reward,
                    confidence))


    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    plt.plot([i for i in range(1, NUM_EPISODES+1)], episode_reward_list, label='Episode reward')
    plt.plot([i for i in range(1, NUM_EPISODES+1)], running_average(
        episode_reward_list, N_EP_RUNNING_AVG), label=f'{N_EP_RUNNING_AVG}-Episode Avg. reward')

    # Corrected method calls for setting labels and title
    plt.xlabel('Episodes')  # Corrected from plt.set_xlabel to plt.xlabel
    plt.ylabel('Total reward')  # Corrected from plt.set_ylabel to plt.ylabel
    plt.title('Total Reward vs Episodes')  # Corrected from plt.set_title to plt.title

    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    agent.save_q_table()


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

    game = SnakeGame(30, 20)
    clock = pygame.time.Clock()
    episode_reward_list = []

    for EPISODE in range(NUM_EPISODES):
        game.reset()
        state, reward, done = game.get_state(), 0, game.done
        action = agent.choose_action(state)
        total_episode_reward = 0.

        while not done:
            next_state, reward, done = game.step(action)
            next_action = agent.choose_action(next_state)
            agent.update_q_table(state, action, reward, next_state, next_action)
            state, action = next_state, next_action

            # Update episode reward
            total_episode_reward += reward

            if RENDER:
                game.render()
                clock.tick(SIMULATION_SPEED)  

            if done:
                print(f"Game Over - Episode {EPISODE} - Reward : {total_episode_reward}")

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
        agent.decay_epsilon()
                
    avg_reward = np.mean(episode_reward_list)
    confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(NUM_EPISODES)


    print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
                    avg_reward,
                    confidence))


    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    plt.plot([i for i in range(1, NUM_EPISODES+1)], episode_reward_list, label='Episode reward')
    plt.plot([i for i in range(1, NUM_EPISODES+1)], running_average(
        episode_reward_list, N_EP_RUNNING_AVG), label=f'{N_EP_RUNNING_AVG}-Episode Avg. reward')

    # Corrected method calls for setting labels and title
    plt.xlabel('Episodes')  # Corrected from plt.set_xlabel to plt.xlabel
    plt.ylabel('Total reward')  # Corrected from plt.set_ylabel to plt.ylabel
    plt.title('Total Reward vs Episodes')  # Corrected from plt.set_title to plt.title

    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    agent.save_q_table()


def DQLearning():
    STATE_SPACE_SIZE = 8
    BATCH_SIZE = 50
    LEARNING_RATE = 0.1
    agent = DQNAgent(  
                    state_space_size=STATE_SPACE_SIZE,
                    action_space_size=ACTION_SPACE_SIZE,
                    alpha=LEARNING_RATE, 
                    gamma=GAMMA, 
                    epsilon=EPSILON,
                    epsilon_decay=EPSILON_DECAY, 
                    epsilon_min=EPSILON_MIN
                )

    game = SnakeGame(30, 20)
    clock = pygame.time.Clock()
    episode_reward_list = []

    for EPISODE in range(NUM_EPISODES):
        game.reset()
        state, reward, done = game.get_state_grid(), 0, game.done
        action = agent.choose_action(state)
        total_episode_reward = 0.

        while not done:
            _ , reward, done = game.step(action)
            next_state = game.get_state_grid()
            next_action = agent.choose_action(next_state)
            agent.remember(state, action, reward, next_state, done)
            agent.learn(BATCH_SIZE)
            state, action = next_state, next_action

            # Update episode reward
            total_episode_reward += reward

            if RENDER:
                game.render()
                clock.tick(SIMULATION_SPEED)  

            if done:
                print(f"Game Over - Episode {EPISODE} - Reward : {total_episode_reward}")

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
                
    avg_reward = np.mean(episode_reward_list)
    confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(NUM_EPISODES)


    print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
                    avg_reward,
                    confidence))


    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    plt.plot([i for i in range(1, NUM_EPISODES+1)], episode_reward_list, label='Episode reward')
    plt.plot([i for i in range(1, NUM_EPISODES+1)], running_average(
        episode_reward_list, N_EP_RUNNING_AVG), label=f'{N_EP_RUNNING_AVG}-Episode Avg. reward')

    # Corrected method calls for setting labels and title
    plt.xlabel('Episodes')  # Corrected from plt.set_xlabel to plt.xlabel
    plt.ylabel('Total reward')  # Corrected from plt.set_ylabel to plt.ylabel
    plt.title('Total Reward vs Episodes')  # Corrected from plt.set_title to plt.title

    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    agent.save()


################################
    
# QLearning()
# Sarsa()
DQLearning()