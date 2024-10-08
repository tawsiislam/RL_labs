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

# Code written by
# Tawsiful Islam, tawsiful@kth.se, 20001110-2035
# Guanyu Lin, guanyul@kth.se, 19980514-5035

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import *
from  ReplayMemory import *
from DDPG_soft_updates import soft_updates
import sys
import pathlib


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

    # agent.saveModel(agent.ActorNet,agent.ActorTarget,fileName_main=ActorNetName,fileName_target=ActorTargetName)
    # agent.saveModel(agent.CriticNet,agent.CriticTarget,fileName_main=CriticNetName,fileName_target=CriticTargetName)

def draw_plots(N_episodes, episode_reward_list, n_ep_running_average, episode_number_of_steps):
    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes for $\gamma = {}$'.format(gamma))
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
    # plt.savefig("plot.png")

def runAgent(env, agent, N):
    env.reset()
    episode_reward_list = []
    episode_number_of_steps = []

    # Training process
    EPISODES = trange(N, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset enviroment data
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0
        while not done:
            # Take a random action
            action = agent.forward(state)
            # Get next state and reward.
            next_state, reward, done, _ = env.step(action)
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

    return episode_reward_list

def agentVrandom(N_episodes,actorNetworkFile: str):
    env = gym.make('LunarLanderContinuous-v2')

    # Random agent initialization
    random_agent=RandomAgent(len(env.action_space.high))
    random_rewards_list = runAgent(env, random_agent, N_episodes)

    # DQN agent initialization
    ddpg_agent = LoadAgent(actorNetworkFile)
    DDPG_rewards_list = runAgent(env, ddpg_agent, N_episodes)
    
    fig = plt.figure(figsize=(9,9))
    episode_list = range(1,N_episodes+1)
    plt.plot(episode_list, random_rewards_list, label="Random Agent")
    plt.plot(episode_list, DDPG_rewards_list, label="DDPG Agent")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Rewards over Episodes between Random & DDPG Agent")
    plt.legend()
    plt.show()

def draw_policy_plots(actorNetworkFile: str, CriticNetworkFile: str):
    ActorNetwork = torch.load(actorNetworkFile)
    QNetwork = torch.load(CriticNetworkFile)
    
    no_h = 100
    no_ang = 100
    
    height_vec = np.double(np.linspace(0, 1.5, no_h))
    angle_vec = np.double(np.linspace(-np.pi, np.pi, no_ang))
    H_mesh, Ang_mesh = np.meshgrid(height_vec, angle_vec)
    Q_tab = np.zeros((no_h,no_ang))
    action_tab = np.zeros((no_h,no_ang))
    
    for hIdx, height in enumerate(height_vec):
        for angIdx, angle in enumerate(angle_vec):
            state = torch.tensor((0,height,0,0,angle,0,0,0), dtype=torch.float32)
            # action = ActorNetwork.forward(state)
            action = ActorNetwork(state)
            action_tab[angIdx, hIdx] = action[1].item()
            Q_tab[angIdx, hIdx] = QNetwork.forward(torch.reshape(state, (1,-1)), torch.reshape(action, (1,-1))).item()
    fig1 = plt.figure()
    ax1 = fig1.gca(projection='3d')
    ax1.plot_surface(Ang_mesh, H_mesh, Q_tab, cmap='viridis')
    ax1.view_init(10,70)
    ax1.set_ylabel("Height $y$")
    ax1.set_xlabel("Angle $\omega$")
    ax1.set_zlabel("Q-value")
    plt.title("Plot of Q-value $Q_{\omega}(s(y,\omega),\pi_{\\theta}(s(y,\omega)))$")
    # plt.savefig("QValue3d.png")
    
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    ax2.plot_surface(Ang_mesh, H_mesh, action_tab, cmap='viridis')
    ax2.view_init(10,120)
    ax2.set_ylabel("Height $y$")
    ax2.set_xlabel("Angle $\omega$")
    ax2.set_zlabel("Best Action")
    plt.title("Plot of best action $\pi_{\\theta}(s(y,\omega)))$")
    # plt.savefig("Action3d.png")
    plt.show()

if __name__ == "__main__":
    # pathName = str(pathlib.Path().resolve())+"\problem2"
    
    # sys.path.insert(0,pathName)
    # Parameters
    N_episodes = 50               # Number of episodes to run for training 300
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
    
    ActorFile = "problem2/neural-network-2-actor.pth"
    CriticFile = "problem2/neural-network-2-critic.pth"
    
    training = False
    plot = False
    compare_plot = True
    print("Starting the script") 
    if training:
        train(N_episodes, gamma, n_ep_running_average, actorLrate, criticLrate, batch_size, buffer_size, tau, d, mu, sigma)
    if plot:
        draw_policy_plots(ActorFile, CriticFile)
    if compare_plot:
        agentVrandom(N_episodes,ActorFile)