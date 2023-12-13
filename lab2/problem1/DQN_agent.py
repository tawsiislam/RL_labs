# Load packages
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random
from collections import deque, namedtuple

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.A(x)

    def save_checkpoint(self):
        print('...saving checkpoint...')
        PATH = "./neural-network-1.pth"
        torch.save(self.state_dict(), PATH)
        
class Agent():
    def __init__(self, n_states, n_actions, batch_size=64, lr=1e-4, gamma=0.99, buffer_size=int(1e5), learn_step=5, tau=1e-3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_step = learn_step
        self.tau = tau

        #model
        self.net_eval = QNetwork(n_states, n_actions)
        self.net_target = QNetwork(n_states, n_actions)
        self.optimizer = optim.Adam(self.net_eval.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        #buffer
        self.memory = ReplayBuffer(n_actions, buffer_size, batch_size)
        self.counter = 0    # update cycle counter

    def choose_action(self, state, epsilon):
        # epsilon-greedy
        if random.random() < epsilon:
            return random.choice(np.arange(self.n_actions))
        
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            #evaluation mode
            self.net_eval.eval()
            with torch.no_grad():
                action_values = self.net_eval(state)
            #back to training
            self.net_eval.train()
            return np.argmax(action_values.data.numpy())
        
    def save2memory(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.counter += 1
        if self.counter % self.learn_step == 0:
            if len(self.memory) >= self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
                
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        q_target = self.net_target(next_states).detach().max(axis=1)[0].unsqueeze(1)
        y_j = rewards + self.gamma * q_target * (1 - dones)          # target, if terminal then y_j = rewards
        q_eval = self.net_eval(states).gather(1, actions)

        # loss backprop
        loss = self.criterion(q_eval, y_j)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update target network
        self.softUpdate()

    def softUpdate(self):
        for eval_param, target_param in zip(self.net_eval.parameters(), self.net_target.parameters()):
            target_param.data.copy_(self.tau*eval_param.data + (1.0-self.tau)*target_param.data)


class ReplayBuffer():
    def __init__(self, n_actions, buffer_size, batch_size):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory = deque(maxlen = buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)