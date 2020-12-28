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
from sklearn.metrics import classification_report, confusion_matrix

k = 10
undersampling = 10_000
train_size = 0.8
n_components = 5
pollution = 0.1
weights = [1,40]

class KNN:
    def __init__(
            self, 
            k, 
            train_X, 
            train_Y, 
            n_components=5, 
            weights=None # must be above 1
            ):
        self.pipeline = make_pipeline(
                StandardScaler(),
                PCA(n_components=n_components)
                ).fit(train_X)
        self.train_X = np.asarray(self.pipeline.transform(train_X))
        self.train_Y = np.asarray(train_Y)
        self.k = k
        self.weights = weights
    
    def distances(self, X):
        X = self.pipeline.transform(X)
        for x in X:
            yield sorted([(np.sum((x-self.train_X[i])**2), i) for i in range(len(self.train_X))])
    
    def k_nearest(self, X):
        for x in self.distances(X):
            yield x[:self.k]
    
    def kth_nearest(self, X):
        for x in self.distances(X):
            yield x[self.k]
    
    def outlier_score(self, X):
        for x in self.kth_nearest(X):
            yield x[0]
    
    def classify(self, X):
        for x in self.k_nearest(X):
            ys = np.asarray(x)[:,1]
            classes = self.train_Y[ys.astype(int)]
            counts = np.bincount(classes.astype(int))
            weighted_counts = [counts[i]*self.weights[i] for i in range(len(counts))]
            yield np.argmax(weighted_counts)
"""
def distance_matrix(self, A, B):
    A = np.asarray(self.pipeline.transform(A))
    train_X = self.train_X
    
    #https://medium.com/swlh/euclidean-distance-matrix-4c3e1378d87f
    p1 = np.sum(test_X**2, axis=1)[:,np.newaxis]
    p2 = np.sum(train_X**2, axis=1)
    p3 = -2 * np.dot(test_X, train_X.T)
    # we are sorting the distances, so taking the square root
    # is not needed
    return p1+p2+p3
"""
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
        pollution=0.5,  # How much of the outliers to put in the training set
        train_size=train_size  # How much of the inliers to put in the training set
        )

knn = KNN(k, train_X, train_Y, n_components, weights=weights)
pred_Y = list(knn.classify(test_X))

print(confusion_matrix(test_Y, pred_Y))
print(classification_report(test_Y, pred_Y))
#plot_report(test_scores, inliers, outliers, 0, 0,xscale="log")
