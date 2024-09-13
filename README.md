# Reinforcement Learning 
This repository contains the lab assignments from the RL course at the Royal Institute of Technology (KTH).
In this repository, there are two labs of interest, lab 1 and lab 2. All the code is implemented in Python. Lab 2 also uses OpenAI Gym library for lab 2's Luna Lander 
and the RL algorithms are implemented together with PyTorch from scratch.

# What is reinforcement learning?
Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with its environment. 
The agent takes action, receives feedback in the form of rewards or penalties, and uses this feedback to improve future decisions. 
Over time, it learns the best actions to maximize rewards. RL has algorithms that require no mathematical model or assumptions for the environment, 
giving a solution well adapted for the real scenario when assumptions cannot be made or when a mathematical model to describe the environment is unknown.

# Lab 1
The link to the report is found [here](https://github.com/tawsiislam/RL_labs/blob/main/lab1/Report/Lin-Guanyu-Islam-Tawsiful-Lab1.pdf).

In this lab, we worked on implementing Q-learning and SARSA with dynamic programming to solve a Maze problem. The agent's objective entails reaching the goal with minimum steps
and avoiding landing on the same square as the minotaur by using action stay (S), up (U), down (D), right (R), and left (L). For the later part of the lab, the agent is required to fetch a key before heading to the goal.
The report discusses for instance how the exploration coefficient for Epsilon-Greedy exploration affects training, how initialisation of the Q-value is the Q-table affects training

The image below shows the policy map for each state/position the agent should take at two different time t based on where the minotaur is located.
Policy Map t=5             |  Policy Map t=8
:-------------------------:|:-------------------------:
![Policy map for each state at time t=5](https://github.com/tawsiislam/RL_labs/blob/main/lab1/images/policy_maze_t5.png) |  ![Policy map for each state at time t=8](https://github.com/tawsiislam/RL_labs/blob/main/lab1/images/policy_maze_t8.png)

# Lab 2
The link to the report is found [here](https://github.com/tawsiislam/RL_labs/blob/main/lab2/Islam-Tawsiful-Lin-Guanyu-Lab2.pdf).

This lab works with the OpenAI's Gym library and the Luna Lander. The agent's goal will be landing the Luna Lander with few steps/actions and it should land
between the two flags. In this lab we implemented Deep Q-network (DQN), Deep Deterministic Policy Gradient (DDPG), and Proximal policy optimization (PPO). 
The report discusses how the training is impacted when using a replay buffer and changing its memory size, changing discount factor gamma, and clipping coefficient 
for PPO adapting how much policy can change.
* When using DQN, the agent had 4 discrete actions: firing down, left, right, or doing nothing. 
* For DDPG, the actions are continuous and clipped between -1 and 1. 
The action determines the angle of the engine and the power depending on the lander's rotational state and height. The actions are deterministic.
* For PPO, the agent controls the angle of the engine and power it will fire but the policy is stochastic and decisions performing certain actions is based on probability

The video below shows the Luna Lander attempting to land between the flags after learning a policy using DQN, DDPG, and PPO.

https://github.com/user-attachments/assets/06d5b4d4-e17c-4921-8e87-f90715964262

