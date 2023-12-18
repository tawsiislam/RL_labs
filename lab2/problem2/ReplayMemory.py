import random
from collections import deque, namedtuple
import torch
import numpy as np


class ReplayMemory:
    def __init__(self, memory_size, batch_size, seed):
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=memory_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        #Add experience
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        #Sample randomly 
        experiences = random.sample(self.memory, k=self.batch_size)

        # convert to torch tensors
        states = torch.from_numpy(np.vstack([experience.state for experience in experiences if experience is not None])).float()
        actions = torch.from_numpy(np.vstack([experience.action for experience in experiences if experience is not None])).long()
        rewards = torch.from_numpy(np.vstack([experience.reward for experience in experiences if experience is not None])).float()
        next_states = torch.from_numpy(np.vstack([experience.next_state for experience in experiences if experience is not None])).float()
        # convert done from boolean to int
        dones = torch.from_numpy(np.vstack([experience.done for experience in experiences if experience is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)