# Written by Guanyu Lin - guanyul@kth.se, and Tawsiful Islam - tawsiful@kth.se
from Model import QNetwork
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ReplayMemory import ReplayMemory


class DQNAgent:
    def __init__(self, state_size, action_size, seed,memory_size,batch_size,update_intervall,tau,gamma):
        # init
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.update_intervall = update_intervall
        self.tau = tau
        self.gamma = gamma
        self.seed = random.seed(seed)
        # initialize Q and Target Q networks
        self.q_network = QNetwork(state_size, action_size, 64)
        self.target_network = QNetwork(state_size, action_size, 64)
        self.optimizer = optim.Adam(self.q_network.parameters())
        # initiliase memory random
        self.memory = ReplayMemory(memory_size, batch_size, seed)
        self.timestep = 0


    def step(self, state, action, reward, next_state, done):
        # update exp
        self.memory.add(state, action, reward, next_state, done)
        self.timestep += 1
        if self.timestep % self.update_intervall == 0:
            if len(self.memory) > self.batch_size:
                sampled_experiences = self.memory.sample()
                self.learn(sampled_experiences)

    def learn(self, experiences):

        #learn from exp
        states, actions, rewards, next_states, dones = experiences

        # get the action with max Q value
        action_values = self.target_network(next_states).detach()
        max_action_values = action_values.max(1)[0].unsqueeze(1)

        # if done just use reward, else update Q_target with discounted action values
        Q_target = rewards + (self.gamma * max_action_values * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)

        # Update θ by performing a backward pass (SGD) on the MSE loss
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # set the target network equal to the main network
        self.update_target_network(self.q_network, self.target_network)

    def update_target_network(self, q_network, target_network):
        #soft update, if tau = 0, it is hard update
        for source_parameters, target_parameters in zip(q_network.parameters(), target_network.parameters()):
            target_parameters.data.copy_(self.tau * source_parameters.data + (1.0 - self.tau) * target_parameters.data)



    def act(self, state, eps=0.0):
        # choose action with epsilon greedy
        rnd = random.random()
        if rnd < eps:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            # set the network into evaluation mode
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            # back to training mode
            self.q_network.train()
            action = np.argmax(action_values.data.numpy())
            return action

    def checkpoint(self, filename):
        torch.save(self.q_network, filename)

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

class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action