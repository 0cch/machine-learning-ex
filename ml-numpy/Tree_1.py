# -*- coding: utf-8 -*-
import pandas as pd
import math
df = pd.read_csv('lenses.txt', sep='\t', header=None)

def CalcShannonEnt(df):
    data_entires = len(df)
    label_count = {}
    label_names = set(df.iloc[:,-1])
    
    for key in label_names:
        label_count[key] = len(df[df.iloc[:,-1]==key])
    
    shannon_ent = 0.0
    for k in label_count:
        prop = float(label_count[k]) / data_entires
        shannon_ent -= prop*math.log(prop,2)
    return shannon_ent

def SplitDataSet(df, i, v):
    return df[df.iloc[:,i]==v].drop(df.columns[i], axis=1)

def ChooseBestFeature(df):
    features_num = df.iloc[0].size-1
    data_entires = len(df)
    best_ent = CalcShannonEnt(df)
    best_feature = -1
    for i in range(features_num):
        label_names = set(df.iloc[:,i])
        new_ent = 0.0
        for v in label_names:
            sub_df = SplitDataSet(df, i, v)
            prop = float(len(sub_df)) / data_entires
            new_ent += prop*CalcShannonEnt(sub_df)
        if new_ent < best_ent:
            best_ent = new_ent
            best_feature = i
    return best_feature

def VoteFeature(df,i):
    feature_vals = set(df.iloc[:,i])
    max_number = 0
    vote_feature = None
    for key in feature_vals:
        cur_number = len(df[df.iloc[:,i]==key])
        if cur_number > max_number:
            max_number = cur_number
            vote_feature = key
    return vote_feature

def CreateTree(df,labels):
    print(df)
    if len(set(df.iloc[:,-1])) == 1:
        return list(set(df.iloc[:,-1]))[0]
    if len(df.columns) == 1:
        return VoteFeature(df,0)
    best_feature = ChooseBestFeature(df)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label:{}}
    del(labels[best_feature])
    feature_vals = set(df.iloc[:,best_feature])
    for key in feature_vals:
        sub_df = SplitDataSet(df, best_feature, key)
        sub_labels = labels.copy()
        my_tree[best_feature_label][key] = CreateTree(sub_df, sub_labels)
    
    return my_tree

labels=[0,1,2,3]
best_feature = ChooseBestFeature(df)
best_feature_label = labels[best_feature]
my_tree = {best_feature_label:{}}
del(labels[best_feature])
feature_vals = set(df.iloc[:,best_feature])













