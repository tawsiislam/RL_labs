# Written by Guanyu Lin - guanyul@kth.se, and Tawsiful Islam - tawsiful@kth.se

import numpy as np
import torch
import matplotlib.pyplot as plt


# load model
model = torch.load('neural-network-1.pth')


# setup limitations
height = np.linspace(0, 1.5, num=100)
angle = np.linspace(-np.pi, np.pi, num=100)
q_values_list = np.zeros((100, 100))

for i in range(100):
    for j in range(100):
        state = [0., height[i], 0., 0., angle[j], 0., 0., 0.]
        q_values = model(torch.tensor([state]).float())
        q_value, action = torch.max(q_values, axis=1)
        q_values_list[j,i] = q_value

## Plot
fig = plt.figure()

height, angle = np.meshgrid(height, angle)

ax = plt.axes(projection='3d')
ax.set_xlabel('Height')
ax.set_ylabel('Angle')
ax.set_zlabel('Q values')
ax.set_title('Q(s, a)')

ax.plot_surface(height, angle, q_values_list, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

plt.show()

