import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from tqdm import trange

from wrappers import apply_wrappers
from agent.utils import Experience
from agent.agent import Agent, SIMPLE, DOUBLE, DUELING, TRAIN, TEST
from utils import running_average


ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL = 5000
NUM_OF_EPISODES = 50_000

BUFFER_LENGHT = 100_000
BATCH_SIZE = 32
C = 128
DISCOUNT_FACTOR = 0.95
N_EP_RUNNING_AVG = 50
MAX_AVG_REWARD = 100
# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(NUM_OF_EPISODES, desc='Episode: ', leave=True)

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)
input_dims = env.observation_space.shape
n_actions = env.action_space.n

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Q-learning agent initialization
my_agent = Agent(input_dims=input_dims , n_actions=n_actions, seed=42, agent_mode=SIMPLE, 
                network_mode=SIMPLE, test_mode=False, batch_size=BATCH_SIZE, n_epochs=1, 
                update_every=C, lr=0.001, gamma=DISCOUNT_FACTOR, eps_min=0.1, eps_max=0.8, 
                Z=1600, max_mem_size=BUFFER_LENGHT, tau=1e-3)

### Training process

for i in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state, _ = env.reset()
    total_episode_reward = 0.
    t = 0
    while not done:

        action = my_agent.choose_action(state)
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, terminated, truncated, _= env.step(action)

        done = truncated or terminated
        #observation = Experience(state, action, reward, next_state, done)
        my_agent.step(state, action, reward, next_state, done)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+= 1

    my_agent.epsilon_decay()        
    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Check if we reach a average rewards of 50
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