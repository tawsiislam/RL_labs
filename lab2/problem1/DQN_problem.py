import random
import sys
from time import time
from tqdm import tqdm
from collections import deque, namedtuple
import numpy as np
import gym
from DQN_agent import DQNAgent
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v2')
env.reset(seed=0)
    
memory_size = int(1e5) # memory size
batch_size = 64    # number of experiences to sample from memory
gamma = 0.99   # discount factor
tau = 1e-3              # soft update parameter, TAU = 0 for hard update
LR = 1e-4               # learning rate
update_intervall = 4        # how often to update Q network
eps = 1

n_episodes = 1000
n_steps = 1000 
Solved_threshhold = 170     

# epsilon  
eps_decay_rate = 0.999 
eps_min = 0.05

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
dqn_agent = DQNAgent(state_size, action_size, seed=0,memory_size=memory_size,batch_size=batch_size,update_intervall=update_intervall,tau=tau,gamma=gamma)

scores = []
recent_scores = []

#List to store 50 recent scores


pbar = tqdm(range(1, n_episodes + 1),unit="ep", ascii=False)
for episode in pbar:
    #env.render()
    state = env.reset()
    score = 0
    for t in range(n_steps):
        action = dqn_agent.act(state, eps)
        next_state, reward, done, info = env.step(action)
        dqn_agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break

        eps = max(eps * eps_decay_rate, eps_min)
    
    # Update the scores_window
    recent_scores.append(score)
    recent_mean_score = np.mean(recent_scores)
    pbar.set_postfix_str(f"Score: {score: 7.2f}, 50 score avg: {recent_mean_score: 7.2f}")

    if recent_mean_score > Solved_threshhold:
        print("Solved!!!!! The Average Score of recent 50 episodes is Above 170 now")
        sys.stdout.flush()
        dqn_agent.checkpoint('neural-network-1.pth')
        break
    
    #pop if there are more than 50 scores in the list
    if len(recent_scores) == 50:
        recent_scores.pop(0)