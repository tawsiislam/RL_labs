# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import *
from  ReplayMemory import *
from DDPG_soft_updates import soft_updates


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

def train(N_episodes, gamma, n_ep_running_average, actorLrate, criticLrate, batchSize, buffer_size, tau, d, mu, sigma):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on",dev)
    
    # Import and initialize Mountain Car Environment
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()
    actionSize = len(env.action_space.high) # dimensionality of the action
    stateSize = len(env.observation_space.high)



    # Reward
    episode_reward_list = []  # Used to save episodes reward
    episode_number_of_steps = []

    # Random Agent initialization
    randomAgent = RandomAgent(actionSize)

    # Buffer initialization
    buffer = ReplayMemory(buffer_size, batch_size)
    while len(buffer) < buffer_size: # Adding random experiences to the buffer
        done = False
        state = env.reset()
        while not done:
            action = randomAgent.forward(state)
            next_state, reward, done, _ = env.step(action)
            buffer.add(state, action, reward, next_state, done)
            state = next_state
    
    # Our agent
    agent = DDPGAgent(dev, stateSize, actionSize, batchSize, actorLrate, criticLrate, mu, sigma, gamma)
    # Training process
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset enviroment data
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0
        while not done:
            # Take a random action
            action = agent.forward(state)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)
            buffer.add(state, action, reward, next_state, done) # Add new experience to the buffer
            
            if len(buffer) > batch_size: 
                agent.backwardCritic(buffer) # Update critic every time
                if t % d == 0:
                    agent.backwardActor(buffer) # Update actor every d steps
                    agent.ActorTarget = soft_updates(agent.ActorNet, agent.ActorTarget, tau)
                    agent.CriticTarget = soft_updates(agent.CriticNet, agent.CriticTarget, tau)
            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t+= 1

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)
        # Close environment
        env.close()

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))
        
    draw_plots(N_episodes, episode_reward_list, n_ep_running_average, episode_number_of_steps)
    CriticNetName = 'CriticNet.pth'
    CriticTargetName = 'CriticTarget.pth'
    ActorNetName = 'ActorNet.pth'
    ActorTargetName = 'ActorTarget.pth'

    agent.saveModel(agent.ActorNet,agent.ActorTarget,fileName_main=ActorNetName,fileName_target=ActorTargetName)
    agent.saveModel(agent.CriticNet,agent.CriticTarget,fileName_main=CriticNetName,fileName_target=CriticTargetName)

def draw_plots(N_episodes, episode_reward_list, n_ep_running_average, episode_number_of_steps):
    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()



if __name__ == "__main__":
    # Parameters
    N_episodes = 300               # Number of episodes to run for training 300
    discount_factor = 0.95         # Value of gamma
    n_ep_running_average = 50      # Running average of 50 episodes
    gamma = 0.99                   # Discount factor
    actorLrate = 5e-5
    criticLrate = 5e-4
    batch_size = 64                # 64
    buffer_size = 30000            # 30000
    tau = 0.001                    # Soft update parameter
    d = 2
    mu = 0.15                      # Noise parameters for normal distribution
    sigma = 0.2
    
    
    training = True
    print("Starting the script") 
    if training:
        train(N_episodes, gamma, n_ep_running_average, actorLrate, criticLrate, batch_size, buffer_size, tau, d, mu, sigma)
