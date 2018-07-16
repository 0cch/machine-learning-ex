# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

df = pd.read_csv('datingTestSet.txt', sep='\t', header=None)
df.loc[df.iloc[:,3] == 'largeDoses', 4] = 3
df.loc[df.iloc[:,3] == 'smallDoses', 4] = 2
df.loc[df.iloc[:,3] == 'didntLike', 4] = 3

X = np.array(df.loc[:,[0,1,2]])
Y = np.array(df.loc[:,4]).reshape((len(Y),1))
def preprocess(d, t):
    return ((d - d.mean(axis=0)) / (d.max(axis=0) - d.min(axis=0)),
            (t - d.mean(axis=0)) / (d.max(axis=0) - d.min(axis=0)))

def kNN(d, t, y, k):
    d,t = preprocess(d,t)
    d -= t
    d = d**2
    d_sum = d.sum(axis=1).reshape(len(d), 1)
    result = np.concatenate((d_sum,y), axis=1)
    result = result[result[:,0].argsort()][0:k]
    return np.array([len(result[result[:,1]==1]),
                     len(result[result[:,1]==2]),
                     len(result[result[:,1]==3])]) / len(result)


    
print(kNN(X,[50000, 10, 2], Y,30))