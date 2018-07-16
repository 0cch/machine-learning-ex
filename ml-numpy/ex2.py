# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
alpha = 0.5
data = np.loadtxt('ex2data1.txt', delimiter=',')
data[:,0:2] = (data[:,0:2] - data[:,0:2].mean(axis=0)) / (data[:,0:2].max(axis=0)-data[:,0:2].min(axis=0))
pos = data[np.where(data[:,2] == 1)]
neg = data[np.where(data[:,2] == 0)]

plt.plot(pos[:,0],pos[:,1],'o',neg[:,0],neg[:,1],'x')

data_rows = len(data)
one_div_m = 1 / data_rows
X = np.concatenate((np.ones((data_rows, 1)), data[:, 0:2]), axis=1)
Y = data[:,2].reshape((data_rows, 1))

theta = np.random.randn(3,1)

for loop_count in range(2000):
    z = X.dot(theta);
    h = sigmoid(z)
    cost = - one_div_m * (Y*np.log(h)+(1-Y)*np.log(1-h)).sum()
    if loop_count % 100 == 0:
        print(cost)
    theta = theta - alpha * one_div_m * np.diagflat(h-Y).dot(X).sum(axis=0).reshape((3,1))
    
X1 = np.linspace(data[:,0].min(), data[:,0].max(), data_rows).reshape((data_rows, 1))
X2 = -(theta[0].repeat(data_rows).reshape((data_rows,1)) + np.diag(theta[1].repeat(data_rows)).dot(X1)) / theta[2]
plt.plot(pos[:,0],pos[:,1],'o',neg[:,0],neg[:,1],'x',X1,X2,'-')