# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data1.txt', delimiter=',');

X = np.concatenate((np.ones((len(data),1)), data[:,0].reshape((len(data),1))), axis=1)
theta = np.random.randn(2,1)
Y = data[:,1].reshape((len(data),1))
alpha = 0.01
div_m = 1 / len(data);

for loop_count in range(1000):
    Y1 = X.dot(theta)
    cost = ((Y-Y1)**2).sum() * 0.5 * div_m;
    print(cost)
    theta = theta - alpha * div_m * np.diagflat(Y1 - Y).dot(X).sum(axis=0).reshape((2,1))


Xl = np.linspace(0, 30, 100)
Yl = Xl * theta[1, 0] + theta[0, 0]

plt.plot(data[:,0],data[:,1],'x', Xl, Yl, 'r')