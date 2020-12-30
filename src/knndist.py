# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 23:06:41 2020

@author: Ask
"""
from preprocessing import get_dataset
from scaler import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from stat_tools import gamma_threshold, split_inliers_outliers
from plotting import plot_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import average_precision_score


# Hyperparameters
k = 10
undersampling = 10_000 #values above 10 000 takes too long to be useful
train_size = 0.4 # KNN works well when undersampling the training data
n_components = 5 # Graphing works poorly for components=2
pollution = 0.1
weights = [1,50] # Outliers are weighted higher than inliers
threshold = 0.99

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
        
        self.train_scores = list(progress_report(self.outlier_score(self.train_X), len(self.train_X)))
        
        self.threshold, self.p = gamma_threshold(self.train_scores, threshold)
        
    
    
    def distances(self, X, transform=True):
        # Transform training data and test data.
        # This will normalize the data such that the variance
        # per axes is made similar.
        if transform:
            X = self.pipeline.transform(X)
        TX = self.pipeline.transform(self.train_X)
        # Calculate the distances between a point in the test data
        # to all the points in hte training data.
        # We are not taking the square root, as we only care about the
        # ordering, and not the specific distances.
        
        # Yield the result per data point in the test data
        # to lower the load on the memory.
        # This allows for bigger datasets without running out of memory
        # as the more effective matrix multiplication method would allow.
        for x in X:
            yield sorted([(np.sum((x-TX[i])**2), i) for i in range(len(TX))])

    def k_nearest(self, X):
        # Yield the k nearest neighbours
        for x in self.distances(X):
            yield x[:self.k]
    
    def kth_nearest(self, X, transform=True):
        # Yield the kth neighbour
        for x in self.distances(X, transform):
            yield x[self.k]
    
    def outlier_score(self, X, transform=True):
        # Yield the score of the kth neightbour.
        # Set a lower bound so that the logarithm does not
        # tend to infinity, as we plot the data on a logarithmic 
        # scale on the x axis.
        for x in self.kth_nearest(X, transform):
            yield max(x[0],1e-2)
    
    
    def os_classify(self, X, transform=True): # unsupervised
        # Classify test data that has a distance score
        # above the threshold as outliers
        for x in self.outlier_score(X,transform):
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
                ys = np.asarray(x)[:,1].astype(int)
                classes = self.train_Y[ys]
                counts = np.bincount(classes)
                # yield the mode class of the neighbours
                yield np.argmax(counts)


def progress_report( stream, length, increments = 10):
    count = 0
    for x in stream:
        count +=1
        if count % (length//increments)==0:
            print(f"{round((count/length)*100)}%")
        yield x

train_X, train_Y, test_X, test_Y = get_dataset(
        sample=undersampling,
        # Not training the encoder on any outliers gives the best results
        pollution=pollution,  # How much of the outliers to put in the training set
        train_size=train_size  # How much of the inliers to put in the training set
        )

knn = KNN(
        k, 
        train_X, 
        train_Y, 
        n_components, 
        weights=weights,
        threshold=0.9
        )
print("predicting outliers based on knn classes")
knn_pred_Y = list(progress_report(knn.classify(test_X),len(test_X)))

print("predicting outliers based on knn outliers scores")
test_knn_outlier_scores = list(progress_report(knn.outlier_score(test_X),len(test_X)))

inlier_scores, outlier_scores = split_inliers_outliers(test_knn_outlier_scores, test_Y)
inliers, outliers = split_inliers_outliers(test_X, test_Y)


knn_os_pred_Y = [1 if score > knn.threshold else 0 for score in test_knn_outlier_scores]

knn_auprc = average_precision_score(test_Y, knn_pred_Y)
knn_os_auprc = average_precision_score(test_Y, knn_os_pred_Y)

baseline = sum(test_Y)/len(test_Y)

print(confusion_matrix(test_Y, knn_pred_Y))
print(classification_report(test_Y, knn_pred_Y))
print(f"AU-PRC: {knn_auprc}")
print(f"baseline: {baseline}")

print(confusion_matrix(test_Y, knn_os_pred_Y))
print(classification_report(test_Y, knn_os_pred_Y))
print(f"AU-PRC: {knn_os_auprc}")
print(f"baseline: {baseline}")

plot_report(knn.train_scores, inlier_scores, outlier_scores, knn.p, knn.threshold, xscale="log", title="KNN outlier scores")


if n_components == 2:
    # https://stackoverflow.com/questions/45075638/graph-k-nn-decision-boundaries-in-matplotlib
    h = 0.1
    X = knn.pipeline.transform(knn.train_X)
    x_min, y_min = -5, -5
    x_max, y_max = 5, 5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = list(knn.os_classify(np.c_[xx.ravel(), yy.ravel()], transform=False))
    Z = np.asarray(Z)
    Z = Z.reshape(xx.shape)
    
    
    transformed_test_X = knn.pipeline.transform(test_X)
    tinliers, toutliers = split_inliers_outliers(transformed_test_X, test_Y)
   
    cmap_light = ListedColormap(['#FFFFAA', '#DDAAFF'])
    plt.figure(figsize=(25,25))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    
    # Plot also the training points
    plt.scatter(tinliers[:, 0], tinliers[:, 1], color="#AAAA00")
    plt.scatter(toutliers[:, 0], toutliers[:, 1], color="purple")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Z1', fontsize=30)
    plt.ylabel('Z2', fontsize=30)
    plt.legend(("inliers", "outliers"), loc="upper right", fontsize=30)
    plt.title("Decission boundary for KNN outlier scoring (pca=2)", fontsize=45)
    plt.show()