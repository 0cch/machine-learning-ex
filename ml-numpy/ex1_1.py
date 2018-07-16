# -*- coding: utf-8 -*-

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data1.txt', delimiter=',');

X = np.concatenate((np.ones((len(data),1)), data[:,0].reshape((len(data),1))), axis=1)
theta = np.random.randn(2,1)
Y = data[:,1].reshape((len(data),1))
alpha = 0.01
div_m = 1 / len(data);
theta_array = np.empty((0, 2))
for loop_count in range(1000):
    Y1 = X.dot(theta)
    cost = ((Y-Y1)**2).sum() * 0.5 * div_m
    if loop_count % 50 == 0:
        print(cost)
    theta = theta - alpha * div_m * np.diagflat(Y1 - Y).dot(X).sum(axis=0).reshape((2,1))
    theta_array = np.append(theta_array, theta.reshape(1,2), axis = 0)

Xl = np.linspace(0, 30, 100)
Yl = Xl * theta[1, 0] + theta[0, 0]

plt.plot(data[:,0],data[:,1],'x', Xl, Yl, 'r')

fig = plt.figure()
#ax = fig.gca(projection='3d')

X3D, Y3D = np.meshgrid(theta_array[:,0], theta_array[:,1])
cost_mat = np.empty((len(X3D), len(X3D)))
Y_real_mat = Y.repeat(len(X3D), axis=1)
for idx in range(len(X3D)):
    theta_3D = np.array([X3D[idx],Y3D[idx]])
    Y_calc_mat = X.dot(theta_3D)
    cost_mat[idx] = ((Y_real_mat - Y_calc_mat) ** 2).sum(axis = 0) * 0.5 * div_m

levels = np.arange(4.4, 5.0, 0.05)
CS = plt.contour(X3D, Y3D, cost_mat, levels=levels)


#surf = ax.plot_surface(X3D, Y3D, cost_mat, cmap = 'jet', 
#                       linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()