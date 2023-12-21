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
# Last update: 26th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LoadAgent(object):
    ''' Agent class for existing .pth model 
    '''
    def __init__(self, actorNetwork):
        self.actor_network = torch.load(actorNetwork)

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation and outputs action according to its policy'''
        action = self.actor_network.forward(torch.tensor(state)).detach().numpy()
        return action

    # def backward(self):
    #     ''' Performs a backward pass on the network '''
    #     pass


class RandomAgent():
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        # super(RandomAgent, self).__init__(n_actions)
        self.n_actions = n_actions

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)
    
class ActorNetwork(nn.Module):
    """ Feedforward actor network designed according to problem instructions"""
    def __init__(self, dev, inputSize, outputSize):
        super().__init__()
        
        layer1Size = 400
        layer2Size = 200
        
        self.input_layer = nn.Linear(inputSize, layer1Size, device=dev)
        self.input_layer_activation = nn.ReLU()
        
        self.hidden_layer_mean = nn.Linear(layer1Size, layer2Size, device=dev) # Calculate mean
        self.hidden_layer_mean_activation = nn.ReLU()
        self.hidden_layer_variance = nn.Linear(layer1Size, layer2Size, device=dev) # Calculate variance
        self.hidden_layer_variance_activation = nn.ReLU()
        
        self.output_layer = nn.Linear(layer2Size, outputSize, device=dev)
        self.output_layer_activation = nn.Tanh()    #Keeps it between -1 and 1
        self.output_layer_var = nn.Linear(layer2Size, outputSize, device=dev)
        self.output_layer_var_activation = nn.Sigmoid()    #Keeps it between -1 and 1
        
    def forward(self, stateTensor: torch.tensor):
        layer1 = self.input_layer(stateTensor)
        layer1Active = self.input_layer_activation(layer1)
        
        layer2_mean = self.hidden_layer_mean(layer1Active)
        layer2Active_mean = self.hidden_layer_mean_activation(layer2_mean)
        layer2_var = self.hidden_layer_variance(layer1Active)
        layer2Active_var = self.hidden_layer_variance_activation(layer2_var)
        
        output_mean = self.output_layer(layer2Active_mean)
        outputActive_mean = self.output_layer_activation(output_mean)
        output_var = self.output_layer(layer2Active_var)
        outputActive_var = self.output_layer_activation(output_var)
        
        return outputActive_mean, outputActive_var
    
class CriticNetwork(nn.Module):
    """ Feedforward critic network designed according to problem instructions"""
    def __init__(self, dev, inputSize, outputSize):
        super().__init__()
        
        layer1Size = 400
        layer2Size = 200
        
        self.input_layer = nn.Linear(inputSize, layer1Size, device=dev)
        self.input_layer_activation = nn.ReLU()
        
        self.hidden_layer = nn.Linear(layer1Size, layer2Size, device=dev)
        self.hidden_layer_activation = nn.ReLU()
        
        self.output_layer = nn.Linear(layer2Size, outputSize, device=dev)
        
    def forward(self, stateTensor: torch.tensor):
        layer1 = self.input_layer(stateTensor)
        layer1Active = self.input_layer_activation(layer1)
        
        layer2 = self.hidden_layer(layer1Active)
        layer2Active = self.hidden_layer_activation(layer2)
        
        output = self.output_layer(layer2Active)
        
        return output
    
class PPOAgent_class(object):
    def __init__(self, dev, stateSize, actionSize, actorLrate, criticLrate, epsilon, gamma):
        self.dev = dev
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.actorLrate = actorLrate
        self.criticLrate = criticLrate
        self.gamma = gamma
        self.epsilon = epsilon
        
        
        criticOutSize = 1
        self.CriticNet = CriticNetwork(self.dev, self.stateSize, criticOutSize)
        
        actorOutSize = self.actionSize # Should be 2 for two dimensional action
        self.ActorNet = ActorNetwork(self.dev, self.stateSize, actorOutSize)
        
        self.OptimCritic = optim.Adam(self.CriticNet.parameters(), lr = self.criticLrate)
        self.OptimActor = optim.Adam(self.ActorNet.parameters(), lr = self.actorLrate)
        
    def forward(self, state: np.ndarray):
        """ Forward computations only done on Actor"""
        mean, var = self.ActorNet(torch.tensor(state, device=self.dev))
        mean = mean.detach().numpy()
        sigma = np.sqrt(var.detach().numpy())
        action = np.random.normal(mean,sigma,size=(2,))
        action = np.clip(action,-1,1)
        return action
    
    def gauss_prob(self, mu, sigma, actions):
        action1Prob = torch.pow(2 * np.pi * sigma[:, 0], -0.5) * torch.exp(-(actions[:, 0] - mu[:, 0]) ** 2 / (2 * sigma[:, 0]))
        action2Prob = torch.pow(2 * np.pi * sigma[:, 1], -0.5) * torch.exp(-(actions[:, 1] - mu[:, 1]) ** 2 / (2 * sigma[:, 1]))
        return action1Prob * action2Prob
            
    
    def backwardCritic(self, buffer, G_i: torch.tensor):
        states, actions, rewards, nextStates, dones = buffer.unzip()
        states = torch.tensor(states, requires_grad=True, device=self.dev)
        actions = torch.tensor(actions, requires_grad=True, device=self.dev)
        self.OptimCritic.zero_grad()
        
        Values = self.CriticNet(states).squeeze()
        
        loss = nn.functional.mse_loss(Values,G_i)
        loss.backward()
        
        nn.utils.clip_grad_norm_(self.CriticNet.parameters(), max_norm=1)
        
        self.OptimCritic.step()
        
        
        
    def backwardActor(self, buffer, G_i: torch.tensor, probOld):
        states, actions, _, _, _ = buffer.unzip()

        self.OptimActor.zero_grad()
        
        valueCritic = self.CriticNet(torch.tensor(states, device=self.dev)).squeeze()
        advantage = G_i - valueCritic
        
        mu_new, var_new = self.ActorNet(torch.tensor(states,requires_grad=True, device=self.dev))
        probNew = self.gauss_prob(mu_new, var_new, torch.tensor(actions,requires_grad=True, device=self.dev))
        ratio = probNew/probOld
        
        value1 = ratio*advantage
        value2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)*advantage
        loss = -torch.mean(torch.min(value1,value2))
        loss.backward()
        
        nn.utils.clip_grad_norm_(self.ActorNet.parameters(), max_norm=1)
        self.OptimActor.step()


    def saveModel(self, mainNet, targetNet, fileName_main='neural-network-main-3.pth',fileName_target='neural-network-target-3.pth'):
        torch.save(mainNet, fileName_main)
        torch.save(targetNet, fileName_target)
        print("Files saved successfully")