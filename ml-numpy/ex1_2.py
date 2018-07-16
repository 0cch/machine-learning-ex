# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data2.txt', delimiter=',')

X = np.concatenate((np.ones((len(data),1)), data[:,0].reshape((len(data),1)) * 0.001, data[:,1].reshape((len(data),1))), axis=1)
theta = np.random.randn(3,1)
Y = data[:,2].reshape((len(data),1)) * 0.00001
alpha = 0.01
div_m = 1 / len(data);
cost_array = []
for loop_count in range(1000):
    Y1 = X.dot(theta)
    cost = ((Y-Y1)**2).sum() * 0.5 * div_m
    if loop_count % 10 == 0:
        cost_array.append(cost)
        print(cost)
    theta = theta - alpha * div_m * np.diagflat(Y1 - Y).dot(X).sum(axis=0).reshape((3,1))

plt.plot(np.arange(0, 1000, 10), cost_array)