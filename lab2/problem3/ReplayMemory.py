import random
from collections import deque, namedtuple
import torch
import numpy as np


class ReplayMemory:
    def __init__(self, memory_size, batch_size):
        self.batch_size = batch_size
        # self.seed = random.seed(seed)
        self.memory = deque(maxlen=memory_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        #Add experience
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        #Sample randomly 
        experiences = np.random.choice(len(self.memory), size=self.batch_size, replace=False)

        batch = [self.memory[experience] for experience in experiences]
        # (states, actions, rewards, next_states, dones) in zip(*batch)
        return zip(*batch)
    
    def unzip(self):
        return zip(*self.memory)

    def __len__(self):
        return len(self.memory)