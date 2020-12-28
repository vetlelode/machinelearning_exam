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
from stat_tools import gamma_threshold, split_inliers_outliers
from plotting import plot_report

k = 10
undersampling = 10_000 #values above 10 000 takes too long to be useful
train_size = 0.8
n_components = 5
pollution = 0.1
weights = [1,40]
threshold = 0.9

class KNN:
    def __init__(
            self, 
            k, 
            train_X, 
            train_Y, 
            n_components=5, 
            weights=[1,1], # must be above 1
            threshold=0.9
            ):
        self.pipeline = make_pipeline(
                StandardScaler(),
                PCA(n_components=n_components)
                ).fit(train_X)
        self.train_X = np.asarray(train_X)
        self.train_Y = np.asarray(train_Y).astype(int)
        self.k = k
        self.weights = weights
        self.train_scores = list(self.outlier_score(self.train_X))
        self.threshold, self.p = gamma_threshold(self.train_scores, threshold)
    
    def distances(self, X, transform=True):
        X = self.pipeline.transform(X)
        TX = self.pipeline.transform(self.train_X)
        for x in X:
            yield sorted([(np.sum((x-TX[i])**2), i) for i in range(len(TX))])
    
    def k_nearest(self, X, transform=True):
        for x in self.distances(X, transform):
            yield x[:self.k]
    
    def kth_nearest(self, X):
        for x in self.distances(X):
            yield x[self.k]
    
    def outlier_score(self, X):
        for x in self.kth_nearest(X):
            yield x[0]
    
    def os_classify(self, X): # unsupervised
        for x in self.outlier_score(X):
            yield 1 if x > self.threshold else 0
    
    def classify(self, X, weighted=True): # supervised
        
        # re-evaluate the distance based off of class weight and distance
        # high weight should lead to higher score
        # low distance should lead to higher score
        weighted_distance = lambda dist, index: self.weights[self.train_Y[index]]/(dist+1)
        
        for x in self.k_nearest(X):
            
            if weighted:
                cl = np.asarray([(weighted_distance(dist, index), self.train_Y[index]) for dist, index in x])
                # yield the class of the neighbour with the highest score
                yield cl[np.argmax(cl[:,0])][1]
            else:
                ys = x[:,1]
                classes = self.train_Y[ys]
                counts = np.bincount(classes)
                # yield the mode class of the neighbours
                yield np.argmax(counts)


train_X, train_Y, test_X, test_Y = get_dataset(
        sample=undersampling,
        # Not training the encoder on any outliers gives the best results
        pollution=pollution,  # How much of the outliers to put in the training set
        train_size=train_size  # How much of the inliers to put in the training set
        )

knn = KNN(k, train_X, train_Y, n_components, weights=weights)
knn_pred_Y = list(knn.classify(test_X))

test_knn_outlier_scores = list(knn.outlier_score(test_X))

inliers, outliers = split_inliers_outliers(test_knn_outlier_scores, test_Y)

knn_os_pred_Y = [1 if score > knn.threshold else 0 for score in test_knn_outlier_scores]

print(confusion_matrix(test_Y, knn_pred_Y))
print(classification_report(test_Y, knn_pred_Y))

print(confusion_matrix(test_Y, knn_os_pred_Y))
print(classification_report(test_Y, knn_os_pred_Y))
plot_report(knn.train_scores, inliers, outliers, knn.p, knn.threshold, xscale="log")
