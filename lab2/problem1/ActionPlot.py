# Written by Guanyu Lin - guanyul@kth.se, and Tawsiful Islam - tawsiful@kth.se

import numpy as np
import torch
import matplotlib.pyplot as plt

# load 
model = torch.load('BasicDQN.pth')

# setup environment
height = np.linspace(0, 1.5, num=15)
angle = np.linspace(-np.pi, np.pi, num=15)

actions_list = np.zeros((15, 15))
for i in range(15):
    for j in range(15):
        state = [0., height[i], 0., 0., angle[j], 0., 0., 0.]
        q_values = model(torch.tensor([state]).float())
        _, action = torch.max(q_values, axis=1)

        actions_list[j, i] = action

        j += 1
    i += 1

fig = plt.figure()

height, angle = np.meshgrid(height, angle)

ax = plt.axes(projection='3d')
ax.set_xlabel('Height')
ax.set_ylabel('Angle')
ax.set_zlabel('action')
ax.set_title('Optimal Action')
ax.set_zticks([0, 1, 2, 3])
ax.set_zlim(0, 3)
ax.scatter(height, angle, actions_list, s=20)


plt.show()