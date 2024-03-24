import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from tqdm import trange
import matplotlib.pyplot as plt

from wrappers import apply_wrappers
from agent.agent import Agent, SIMPLE, DOUBLE, DUELING, TRAIN, TEST
from utils import running_average


ENV_NAME = 'SuperMarioBros-1-1-v3'
DISPLAY = False

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', 
                                apply_api_compatibility=True)

env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

INPUT_DIMS = env.observation_space.shape
N_ACTIONS = env.action_space.n

NUM_OF_EPISODES = 64_000
NUM_OF_EP_DECAY = int(NUM_OF_EPISODES*0.8)
NB_EPOCHS = 3
BUFFER_LENGHT = 128_000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95
EPS_MIN = 0.05
EPS_MAX = 0.95

N_EP_RUNNING_AVG = 50
MAX_AVG_REWARD = 100

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(NUM_OF_EPISODES, desc='Episode: ', leave=True)

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Q-learning agent initialization
my_agent = Agent(input_dims=INPUT_DIMS, 
                 n_actions=N_ACTIONS,
                 max_mem_size=BUFFER_LENGHT, 
                 Z=NUM_OF_EP_DECAY,
                 agent_mode=SIMPLE, 
                 network_mode=SIMPLE, 
                 batch_size=BATCH_SIZE, 
                 n_epochs=NB_EPOCHS, 
                 lr=LEARNING_RATE, 
                 gamma=DISCOUNT_FACTOR, 
                 eps_min=EPS_MIN, 
                 eps_max=EPS_MAX)

### Training process

for i in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state, _ = env.reset()
    total_episode_reward = 0.
    t = 0
    while not done:
        #env.render()
        action = my_agent.choose_action(state)
        next_state, reward, terminated, truncated, _= env.step(action)
        done = truncated or terminated
        my_agent.step(state, action, reward, next_state, done)
        # Update episode reward
        total_episode_reward += reward
        # Update state for next iteration
        state = next_state
        t += 1

    
    my_agent.epsilon_decay()
    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # # Check if we reach a average rewards of 50
    # if running_average(episode_reward_list, N_EP_RUNNING_AVG)[-1] >= MAX_AVG_REWARD:
    #     my_agent.save_model()
    #     MAX_AVG_REWARD += 10 
    #     #break

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, N_EP_RUNNING_AVG)[-1],
        running_average(episode_number_of_steps, N_EP_RUNNING_AVG)[-1]))
    
# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, NUM_OF_EPISODES+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, NUM_OF_EPISODES+1)], running_average(
    episode_reward_list, N_EP_RUNNING_AVG), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, NUM_OF_EPISODES+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, NUM_OF_EPISODES+1)], running_average(
    episode_number_of_steps, N_EP_RUNNING_AVG), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()