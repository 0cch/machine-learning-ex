# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
alpha = 0.5
lamb = 0.001
data = np.loadtxt('ex2data2.txt', delimiter=',')
data[:,0:2] = (data[:,0:2] - data[:,0:2].mean(axis=0)) / (data[:,0:2].max(axis=0)-data[:,0:2].min(axis=0))
pos = data[np.where(data[:,2] == 1)]
neg = data[np.where(data[:,2] == 0)]

plt.plot(pos[:,0],pos[:,1],'o',neg[:,0],neg[:,1],'x')
data_rows = len(data)
one_div_m = 1 / data_rows
X = np.concatenate((np.ones((data_rows, 1)), data[:, 0:2], 
                    data[:, 0].reshape((data_rows, 1))*data[:, 1].reshape((data_rows, 1)),
                    data[:, 0].reshape((data_rows, 1))**2, 
                    data[:, 1].reshape((data_rows, 1))**2), axis=1)
Y = data[:,2].reshape((data_rows, 1))

theta = np.random.randn(6,1)

for loop_count in range(2000):
    z = X.dot(theta);
    h = sigmoid(z)
    cost = - one_div_m * (Y*np.log(h)+(1-Y)*np.log(1-h)).sum() + lamb*one_div_m*0.5*(theta**2).sum() 
    if loop_count % 100 == 0:
        print(cost)
    theta = (theta - alpha * one_div_m * np.diagflat(h-Y).dot(X).sum(axis=0).reshape((6,1)) - 
             lamb*one_div_m*np.concatenate(([[0]], theta[1:])))
    
X1 = np.linspace(data[:,0].min(), data[:,0].max(), data_rows).reshape((data_rows, 1))
X2 = np.linspace(data[:,1].min(), data[:,1].max(), data_rows).reshape((data_rows, 1))

XC, YC = np.meshgrid(X1, X2)
Z = theta[0]+theta[1]*XC+theta[2]*YC+theta[3]*XC*YC+theta[4]*XC**2+theta[5]*YC**2
H = sigmoid(Z)

plt.contour(XC, YC, Z, [0])