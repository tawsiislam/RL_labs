# Written by Guanyu Lin - guanyul@kth.se, and Tawsiful Islam - tawsiful@kth.se
import numpy as np
import gym
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from DQN_agent import RandomAgent

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

# Load model
try:
    model = torch.load('BasicDQN.pth')
    print('Network model: {}'.format(model))
except:
    print('File neural-network-1.pth not found!')
    exit(-1)

# Import and initialize Mountain Car Environment
env = gym.make('LunarLander-v2')
env.reset()
env1 = gym.make('LunarLander-v2')
env1.reset()

# Parameters
N_EPISODES = 50            # Number of episodes to run for trainings
CONFIDENCE_PASS = 50

# Reward
n_actions = env1.action_space.n               # Number of available actions
dim_state = len(env1.observation_space.high)  # State dimensionality
episode_reward_list = []  # Used to store episodes reward
episode_reward_list1 = []
agent = RandomAgent(n_actions)

# Simulate episodes
EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
for i in EPISODES:
    EPISODES.set_description("Episode {}".format(i))
    # Reset enviroment data
    done = False
    state = env.reset()
    total_episode_reward = 0.
    while not done:
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        q_values = model(torch.tensor([state]))
        _, action = torch.max(q_values, axis=1)
        next_state, reward, done, _ = env.step(action.item())

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state

    # Append episode reward
    episode_reward_list.append(total_episode_reward)

    # Close environment
    env.close()

EPISODES1 = trange(N_EPISODES, desc='Episode: ', leave=True)
for i in EPISODES1:
    EPISODES1.set_description("Episode {}".format(i))
    # Reset enviroment data
    done = False
    state = env1.reset()
    total_episode_reward = 0.
    while not done:
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        action = agent.forward(state)
        next_state, reward, done, _ = env1.step(action)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state

    # Append episode reward
    episode_reward_list1.append(total_episode_reward)

    # Close environment
    env1.close()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, 51)], episode_reward_list, label='Episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, 51)], episode_reward_list1, label='Steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()

avg_reward = np.mean(episode_reward_list)
confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)


print('Your Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
                avg_reward,
                confidence))

avg_reward = np.mean(episode_reward_list1)
confidence = np.std(episode_reward_list1) * 1.96 / np.sqrt(N_EPISODES)


print('Random Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
                avg_reward,
                confidence))