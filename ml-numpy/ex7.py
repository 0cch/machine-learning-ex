# -*- coding: utf-8 -*-
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

X = sio.loadmat('ex7data2.mat')['X']
#plt.plot(X[:,0],X[:,1],'.')
m,n = X.shape
class_number = 3

random_index = np.random.randint(0, m, class_number)
center_pos = X[random_index]
class_map = {}

for loop_count in range(100):
    distance = np.empty((m,0))
    
    for pos in center_pos:
        XX = ((X-pos)**2).sum(axis=1).reshape((m,1))
        distance = np.append(distance, XX, axis=1)
    
    class_index = np.argmin(distance, axis=1)
    
    for i in range(class_number):
        center_pos[i] = X[np.where(class_index==i)].mean(axis=0)
        class_map[i] = X[np.where(class_index==i)]

plt.plot(class_map[0][:,0],class_map[0][:,1],'.',
         class_map[1][:,0],class_map[1][:,1],'.',
         class_map[2][:,0],class_map[2][:,1],'.')