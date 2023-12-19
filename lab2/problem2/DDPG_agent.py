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


class Agent(object):
    ''' Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    '''
    def __init__(self, actorNetwork):
        self.actor_network = torch.load(actorNetwork)

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        action = self.actor_network.forward(torch.tensor(state)).detach().numpy()
        return action

    # def backward(self):
    #     ''' Performs a backward pass on the network '''
    #     pass


class RandomAgent(Agent):
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
        
        self.hidden_layer = nn.Linear(layer1Size, layer2Size, device=dev)
        self.hidden_layer_activation = nn.ReLU()
        
        self.output_layer = nn.Linear(layer2Size, outputSize, device=dev)
        self.output_layer_activation = nn.Tanh()    #Keeps it between -1 and 1
        
    def forward(self, stateTensor: torch.tensor):
        layer1 = self.input_layer(stateTensor)
        layer1Active = self.input_layer_activation(layer1)
        
        layer2 = self.hidden_layer(layer1Active)
        layer2Active = self.hidden_layer_activation(layer2)
        
        output = self.output_layer(layer2Active)
        outputActive = self.output_layer_activation(output)
        
        return outputActive
    
class CriticNetwork(nn.Module):
    """ Feedforward critic network designed according to problem instructions"""
    def __init__(self, dev, inputSize, outputSize, actionSize):
        super().__init__()
        
        layer1Size = 400
        layer2Size = 200
        
        self.input_layer = nn.Linear(inputSize, layer1Size, device=dev)
        self.input_layer_activation = nn.ReLU()
        
        self.hidden_layer = nn.Linear(layer1Size+actionSize, layer2Size, device=dev)
        self.hidden_layer_activation = nn.ReLU()
        
        self.output_layer = nn.Linear(layer2Size, outputSize, device=dev)
        
    def forward(self, stateTensor: torch.tensor, actionTensor):
        layer1 = self.input_layer(stateTensor)
        layer1Active = self.input_layer_activation(layer1)
        
        concat = torch.cat([layer1Active, actionTensor], dim = 1)
        layer2 = self.hidden_layer(concat)
        layer2Active = self.hidden_layer_activation(layer2)
        
        output = self.output_layer(layer2Active)
        
        return output
    
class DDPGAgent(object):
    def __init__(self, dev, stateSize, actionSize, batchSize, actorLrate, criticLrate, mu, sigma, gamma):
        self.dev = dev
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.batchSize = batchSize
        self.actorLrate = actorLrate
        self.criticLrate = criticLrate
        self.gamma = gamma
        self.mu = mu
        self.sigma = sigma
        self.oldNoise = np.array([0,0])
        
        
        criticOutSize = 1
        self.CriticNet = CriticNetwork(self.dev, self.stateSize, criticOutSize, self.actionSize)
        self.CriticTarget = CriticNetwork(self.dev, self.stateSize, criticOutSize, self.actionSize)
        
        actorOutSize = 2
        self.ActorNet = ActorNetwork(self.dev, self.stateSize, actorOutSize)
        self.ActorTarget = ActorNetwork(self.dev, self.stateSize, actorOutSize)
        
        self.OptimCritic = optim.Adam(self.CriticNet.parameters(), lr = self.criticLrate)
        self.OptimActor = optim.Adam(self.ActorNet.parameters(), lr = self.actorLrate)
        
    def forward(self, state: np.ndarray):
        w = np.random.normal(0, self.sigma, size=2)
        noise = -self.mu*self.oldNoise + w
        a = self.ActorNet.forward(torch.tensor(state, device=self.dev)).detach().cpu().numpy() + noise
        self.oldNoise = noise
        a = np.clip(a,-1,1).reshape(-1)
        return a
    
    def backwardCritic(self, bufferExperiences):
        states, actions, rewards, nextStates, dones = bufferExperiences.sample()
        
        self.OptimCritic.zero_grad()
        
        with torch.no_grad():
            targetNextAction = self.ActorTarget.forward(torch.tensor(nextStates, device=self.dev))
            targetQValue = self.CriticTarget.forward(torch.tensor(nextStates, device=self.dev),targetNextAction)
            yValue = (torch.tensor(rewards, device=self.dev, dtype=torch.float32)) \
            + (self.gamma*targetQValue.squeeze())*(torch.tensor(dones, device=self.dev)==False)
            
        states = torch.tensor(states, device=self.dev,requires_grad=True, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.dev,requires_grad=True, dtype=torch.float32)
        Qvalues = self.CriticNet.forward(states, actions).squeeze()
        
        loss = nn.functional.mse_loss(Qvalues,yValue)
        loss.backward()
        
        nn.utils.clip_grad_norm_(self.CriticNet.parameters(), max_norm=1.)
        
        self.OptimCritic.step()
        
    def backwardActor(self, bufferExperiences):
        states, actions, _, _, _ = bufferExperiences.sample()
        
        self.OptimActor.zero_grad()
        states = torch.tensor(states, device=self.dev,requires_grad=True, dtype=torch.float32)
        actions = self.ActorNet.forward(states)
        Qvalues = self.CriticNet.forward(states,actions).squeeze()
        
        loss = -torch.mean(Qvalues)
        loss.backward()
        
        nn.utils.clip_grad_norm_(self.ActorNet.parameters(), max_norm=1.)
        self.OptimActor.step()

    def saveModel(self, mainNet, targetNet, fileName_main='neural-network-main-2.pth',fileName_target='neural-network-target-2.pth'):
        torch.save(mainNet, fileName_main)
        torch.save(targetNet, fileName_target)
        print("Files saved successfully")