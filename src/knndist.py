# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 23:06:41 2020

@author: Ask
"""
from preprocessing import get_dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

k = 2
undersampling = 1000
train_size = 0.5
n_components = 5

class KNN_outlier_factor:
    def __init__(self, k, train_X, n_components=5):
        self.pipeline = make_pipeline(
                StandardScaler(),
                PCA(n_components=n_components)
                ).fit(train_X)
        self.train_X = np.asarray(self.pipeline.transform(train_X))
        self.k = k
    
    def kdist(self, test_X):
        test_X = np.asarray(self.pipeline.transform(test_X))
        k_dist = []
        for test_x in test_X:
            neighbours = [np.sum((test_x-train_x)**2)**0.5 for train_x in self.train_X]
            # Only consider the distance to the k-th nearest neighbour.
            k_dist.append(sorted(neighbours)[self.k])
        return k_dist
    

def split_inliers_outliers(X, Y):
    inliers, outliers = [], []
    for i in range(len(X)):
        if Y[i] == 0:
            inliers.append(X[i])
        else:
            outliers.append(X[i])
    return inliers, outliers


train_X, train_Y, test_X, test_Y = get_dataset(
        sample=undersampling,
        # Not training the encoder on any outliers gives the best results
        pollution=0,  # How much of the outliers to put in the training set
        train_size=train_size  # How much of the inliers to put in the training set
        )

knnof = KNN_outlier_factor(k, train_X, n_components)
train_scores = np.asarray(knnof.kdist(train_X))+1
test_scores = np.asarray(knnof.kdist(test_X))+1
inliers, outliers = split_inliers_outliers(test_scores,test_Y)

#plot_report(test_scores, inliers, outliers, 0, 0,xscale="log")
