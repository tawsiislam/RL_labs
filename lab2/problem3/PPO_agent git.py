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
# Last update: 29th October 2020, by alessior@kth.se
#

# Load packages
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

class Agent(object):
    ''' Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    '''
    def __init__(self, actor_network):
        self.main_actor = torch.load(actor_network)

    def forward(self, state):
        mu, sig = self.main_actor(torch.tensor(state, device=self.dev))    # mu and sigma are tensors of the same dimensionality of the action
        mu = mu.detach().cpu().numpy()
        std = np.sqrt(sig.detach().cpu().numpy())
        action_1 = np.random.normal(mu[0], std[0])
        action_2 = np.random.normal(mu[1], std[1])
        actions = np.clip([action_1, action_2], -1, 1)
        return actions

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent():
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int, actor_network):
        self.n_actions = n_actions

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)

class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """
    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        #return batch
        return zip(*batch)

    def unzip_buffer(self):
        return zip(*self.buffer)


class ActorNetwork(nn.Module): #Actor is policy
    """ Actor feedforward neural network """
    def __init__(self, dev, input_size, output_size):
        super().__init__()

        #Define number of neurons: given by the exercise
        num_neurons_l1 = 400
        num_neurons_l2 = 200

        # INPUT: state
        self.input_layer = nn.Linear(input_size, num_neurons_l1, device=dev)
        self.input_layer_activation = nn.ReLU()

        # HIDEN + OUTPUT LAYER MEAN
        self.hidden_layer_mean = nn.Linear(num_neurons_l1, num_neurons_l2, device=dev)
        self.hidden_layer_activation_mean = nn.ReLU()
        self.output_layer_mean = nn.Linear(num_neurons_l2, output_size, device=dev)
        self.output_layer_activation_mean = nn.Tanh()

        # HIDEN + OUTPUT LAYER VARIANCE
        self.hidden_layer_var = nn.Linear(num_neurons_l1, num_neurons_l2, device=dev)
        self.hidden_layer_activation_var = nn.ReLU()
        self.output_layer_var = nn.Linear(num_neurons_l2, output_size, device=dev)
        self.output_layer_activation_var = nn.Sigmoid()


    def forward(self, x):
        # Computation policy(s)

        # Compute first layer
        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        # HIDEN LAYER MEAN
        l2_mean = self.hidden_layer_mean(l1)
        l2_mean = self.hidden_layer_activation_mean(l2_mean)
        out_mean = self.output_layer_mean(l2_mean)
        out_mean = self.output_layer_activation_mean(out_mean)

        # HIDEN LAYER VAR
        l2_var = self.hidden_layer_var(l1)
        l2_var = self.hidden_layer_activation_var(l2_var)
        out_var = self.output_layer_var(l2_var)
        out_var = self.output_layer_activation_var(out_var)

        return out_mean, out_var



class CriticNetwork(nn.Module): #Critic is Q
    """ Critic feedforward neural network """
    def __init__(self, dev, input_size, output_size=1):
        super().__init__()
        # Define number of neurons: given by the exercise
        num_neurons_l1 = 400
        num_neurons_l2 = 200

        # INPUT: state
        self.input_layer = nn.Linear(input_size, num_neurons_l1, device=dev)
        self.input_layer_activation = nn.ReLU()

        # Concatenate action
        self.hidden_layer = nn.Linear(num_neurons_l1, num_neurons_l2, device=dev)
        self.hidden_layer_activation = nn.ReLU()

        # OUTPUT: Q(s,a) (1 dimension)
        self.output_layer = nn.Linear(num_neurons_l2, output_size, device=dev)

    def forward(self, x):
        # Computation of Q(s,a)

        # Compute first layer
        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        l2 = self.hidden_layer(l1)
        l2 = self.hidden_layer_activation(l2)

        # Compute output layer
        out = self.output_layer(l2)

        return out



class PPOAgent(object):
    ''' Base agent class'''

    def __init__(self, discount_factor, lr_actor, lr_critic, action_size, dim_state, epsilon, dev):
        #Parameters
        self.dev = dev

        self.discount_factor = discount_factor
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.dim_state = dim_state
        self.action_size = action_size
        self.epsilon = epsilon

        #Critic networks (Q(s,a))
        self.main_critic = CriticNetwork(self.dev, dim_state)

        #Actor networks (policy(s))
        self.main_actor = ActorNetwork(self.dev, dim_state, action_size)  #main ANN: ANN to update in every batch size

        #OPITMIZER
        self.optimizer_critic = optim.Adam(self.main_critic.parameters(), lr=self.lr_critic) #I put the optimizer on the main_ann 'cause is the one we are gonna train
        self.optimizer_actor = optim.Adam(self.main_actor.parameters(), lr=self.lr_actor)

    def forward(self, state):
        mu, sig = self.main_actor(torch.tensor(state, device=self.dev))    # mu and sigma are tensors of the same dimensionality of the action
        mu = mu.detach().cpu().numpy()
        std = np.sqrt(sig.detach().cpu().numpy())
        action_1 = np.random.normal(mu[0], std[0])
        action_2 = np.random.normal(mu[1], std[1])
        actions = np.clip([action_1, action_2], -1, 1)
        return actions

    def gaussian_probability(self, mu, sig, actions):
        "Function to compute pi_parameter"
        prob_action_1 = torch.pow(2 * np.pi * sig[:, 0], -0.5) * torch.exp(-(actions[:, 0] - mu[:, 0]) ** 2 / (2 * sig[:, 0]))
        prob_action_2 = torch.pow(2 * np.pi * sig[:, 1], -0.5) * torch.exp(-(actions[:, 1] - mu[:, 1]) ** 2 / (2 * sig[:, 1]))
        final_prob = prob_action_1 * prob_action_2
        return final_prob

    def update(self, buffer, M):
        '''Train agent'''
        state, action, reward, next_state, done = buffer.unzip_buffer()
        time_step = len(state)

        # Compute the target value G_i for ech (s_i,a_i)
        g_i = np.zeros((time_step))
        g_i[-1] = reward[-1] #*discount_factor: first time step == 1
        for t in reversed(range(time_step-1)): # We start at time_step - 1
            g_i[t]= g_i[t+1]*self.discount_factor+reward[t]
        g_i = torch.tensor(g_i, dtype=torch.float32, device=self.dev) #we need to convert it to a tensor

        # Calculate V_w for each (S_i) to Build the advantadge estimator
        # We need to set the gradients to true for pytorch to calculate the gradients
        state_grad = torch.tensor(state, requires_grad = True, device=self.dev)
        actions = torch.tensor(action, requires_grad = True, device=self.dev)

        #Old fix    prob under pi
        old_mean, old_var = self.main_actor(state_grad)
        old_prob = self.gaussian_probability(old_mean,old_var,actions).detach()

        #Over m iterations:
        #Compute Advanradge Estimator over m iterations
        for m in range(M):
            # Update critic network
            # Set gradient to zero
            self.optimizer_critic.zero_grad()

            # Compute outpute (value approx from critic) [why here they use gradient?]
            y = self.main_critic(state_grad).squeeze()

            # Compute MSE loss
            loss = nn.functional.mse_loss(y, g_i)

            # Comput gradient
            loss.backward()

            nn.utils.clip_grad_norm_(self.main_critic.parameters(), max_norm=1)

            # Perform backward pass (backpropagation)
            self.optimizer_critic.step()

            # Update actor network
            # Set gradient to zero: RECOMENDATION: Always set gradient to zero inpytorch and the begining of the training
            self.optimizer_actor.zero_grad()

            # Compute output (value function) for critic
            y_without_grad = self.main_critic(torch.tensor(state, device=self.dev)).squeeze()

            # Advantadge function
            advantadge = g_i - y_without_grad

            # Calculate ratio between old/new prob
            new_mean, new_var = self.main_actor(state_grad)
            new_prob = self.gaussian_probability(new_mean, new_var, actions)
            ratio = new_prob/old_prob

            # Calculate loss
            first_value = ratio*advantadge
            second_value = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantadge
            loss = torch.min(first_value, second_value)
            loss = -torch.mean(loss) #how Aleix taught us!

            # Comput gradient
            loss.backward()

            nn.utils.clip_grad_norm_(self.main_actor.parameters(), max_norm=1)

            # Perform backward pass (backpropagation)
            self.optimizer_actor.step()



    def save_ann(self, actor_nn, critic_nn, filename_actor='actor-neural-network-main-1.pth',filename_critic='critic-network-target-1.pth'):
        '''Save network in working directory'''
        torch.save(actor_nn, filename_actor)
        print(f'Saved main_network as {filename_actor}')
        torch.save(critic_nn, filename_critic)
        print(f'Saved main_network as {filename_critic}')
        return