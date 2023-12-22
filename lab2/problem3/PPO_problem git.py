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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#


# Exercise solution by:
# Clara Escorihuela Altaba - 19980504-T283
# Joana Palés Huix - 19970213-4629


# Load packages
import numpy as np
import gym
import torch
from tqdm import trange
from PPO_agent import *
import matplotlib as mpl
import matplotlib.pyplot as plt

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


def test_agent(env, agent, N):
    """ Let the agent behave with the policy it follows"""
    env.reset()

    episode_reward_list = []
    episode_number_of_steps = []

    EPISODES = trange(N, desc='Episodes', leave=True)
    for i in EPISODES:
        done = False
        state = env.reset()
        total_episode_reward = 0
        t = 0

        while not done:
            action = agent.forward(state)
            # Get next state, reward and done. Append into a buffer
            next_state, reward, done, _ = env.step(action)
            # Update episode reward
            total_episode_reward += reward
            # Update state for next iteration
            state = next_state
            t += 1

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Close environment
        env.close()

    return N, episode_reward_list, episode_number_of_steps

def compare_to_random(n_episodes, network_filename = 'neural-network-3-actor.pth'):
    """ Compare random to train agnet, trained agent uses the neural
    network with weights uploaded in network_filename
    """
    env = gym.make('LunarLanderContinuous-v2')

    # Random agent initialization
    random_agent=RandomAgent(len(env.action_space.high), network_filename)
    num_episodes_random,random_rewards,_ = test_agent(env, random_agent, n_episodes)

    # DQN agent initialization
    ppo_agent = Agent(network_filename)
    num_episodes_dqn,dqn_rewards,_ = test_agent(env, ppo_agent, n_episodes)

    # Plot rewads
    fig = plt.figure(figsize=(9, 9))

    xr = [i for i in range(1, num_episodes_random+1)]
    xdqn = [i for i in range(1, num_episodes_dqn+1)]
    plt.plot(xr, random_rewards, label='Random Agent')
    plt.plot(xdqn, dqn_rewards, label='Trained DQN Agent')
    plt.ylim(-400, 400)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward vs Episodes')
    plt.legend()
    plt.show()

def train(N_episodes, discount_factor, n_ep_running_average, lr_actor, lr_critic, buffer_size, epsilon, epochs, legend_main_actor,legend_main_critic ):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(dev)

    # Import and initialize Mountain Car Environment
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()
    action_size = len(env.action_space.high)  # dimensionality of the action
    dim_state = len(env.observation_space.high)  # State dimensionality

    # Reward
    episode_reward_list = []  # this list contains the total reward per episode
    episode_number_of_steps = []  # this list contains the number of steps per episode

    # Training process
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    # DQN agent initialization
    agent = PPOAgent(discount_factor, lr_actor, lr_critic, action_size, dim_state, epsilon, dev)

    for i in EPISODES:
        # Reset environment data
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0

        # Buffer initialization
        buffer = ExperienceReplayBuffer(maximum_length=buffer_size)

        while not done:
            # Take a random action
            action = agent.forward(state) #two dim: [main engine, fire left/right]

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))

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

        agent.update(buffer,epochs)

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))

    draw_plots(N_episodes, episode_reward_list, n_ep_running_average, episode_number_of_steps)
    agent.save_ann(agent.main_actor,agent.main_critic,filename_actor=legend_main_actor,filename_critic=legend_main_critic)



def draw_plots(N_episodes, episode_reward_list, n_ep_running_average, episode_number_of_steps):
    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
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




def optimal_policy_plot(actor_network= 'neural-network-3-actor.phd', critic_network= 'neural-network-3-critic.phd'):
    n_y = 100
    n_om = 100
    ys = np.linspace(0, 1.5, n_y)
    ws = np.linspace(-np.pi, np.pi, n_om)

    Ys, Ws = np.meshgrid(ys, ws)

    policy_network = torch.load(actor_network)
    V_network = torch.load(critic_network)

    V = np.zeros((len(ys), len(ws)))
    mu = np.zeros((len(ys), len(ws)))
    for y_idx, y in enumerate(ys):
        for w_idx, w in enumerate(ws):
            state = torch.tensor((0, y, 0, 0, w, 0, 0, 0), dtype=torch.float32)
            a = policy_network(state)
            mu[w_idx, y_idx] = a[0][1].item()
            V[w_idx, y_idx] = V_network(torch.reshape(state, (1,-1))).item()

    #3d plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(Ws, Ys, V, cmap=mpl.cm.coolwarm)
    ax.set_ylabel('height (y)')
    ax.set_xlabel('angle (ω)')
    ax.set_zlabel('V(s(y,ω))')
    plt.show()

    # 3d plot
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    surf2 = ax2.plot_surface(Ws, Ys, mu)
    ax2.set_ylabel('height (y)')
    ax2.set_xlabel('angle (ω)')
    ax2.set_zlabel('μ(s,ω)')
    plt.show()

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    im = ax3.pcolormesh(Ys, Ws, mu)
    cbar = fig3.colorbar(im, ticks=[0, 1, 2, 3])
    ax3.set_ylabel('angle (ω)')
    ax3.set_xlabel('height (y)')
    #cbar.ax3.set_yticklabels(['nothing (0)', 'left (1)', 'main (2)', 'right (3)'])
    ax3.set_title('μ(s,ω)')
    plt.show()



if __name__ == "__main__":

    n_episodes = 1600  # Number of episodes - 50 for comparative
    discount_factor = 0.99  # Value of gamma
    n_ep_running_average = 50  # Running average of 50 episodes
    lr_actor = 1e-5  # For actor model
    lr_critic = 1e-3  # For actor model
    buffer_size = 30000  # L
    d = 2
    epochs = 10 #M
    epsilon = 0.2

    train_ex = True
    comparative = False
    actor_filename = 'neural-network-3-actor.pth.pth'
    critic_filename = 'neural-network-3-critic.pth'
    plot_3d = False

    if train_ex == True:
        train(n_episodes, discount_factor, n_ep_running_average, lr_actor, lr_critic, buffer_size, epsilon, epochs, actor_filename, critic_filename)

    if comparative == True:
        compare_to_random(n_episodes, network_filename=actor_filename)

    if plot_3d == True:
        optimal_policy_plot(actor_network = actor_filename, critic_network = critic_filename)