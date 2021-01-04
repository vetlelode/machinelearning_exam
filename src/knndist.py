# -*- coding: utf-8 -*-
from preprocessing import get_dataset
from scaler import StandardScaler
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from stat_tools import gamma_threshold, split_inliers_outliers
from stat_tools import OutlierDetectorScorer
from plotting import plot_report, prc_plot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import average_precision_score

from sklearn.neighbors import BallTree

# Hyperparameters
k = 20
undersampling = 72000 #values above 10 000 takes too long to be useful
train_size = 1-0.9722222222222222# KNN works well when undersampling the training data
n_components = 5 # Graphing works poorly for components=2
pollution = 0.05
weights = [1,1] # best weights are surprisingly enough 1:1
threshold = 0.99

iterations = 10

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
        
        self.train_X = np.asarray(train_X)
        self.tree = BallTree(self.train_X)

        self.train_Y = np.asarray(train_Y).astype(int)
        self.k = k
        self.weights = weights
        
        print("scoring training data")
        self.train_scores = self.outlier_score(self.train_X)
        
        #self.train_scores = list(progress_report(self.outlier_score(self.train_X), len(self.train_X)))
        print("thresholding training scores")
        self.threshold, self.p = gamma_threshold(self.train_scores, threshold)
        
    def outlier_score(self, X):
        dist, indices = self.query(X)
        return np.maximum(dist[:,self.k-1],1e-2)
    
    def query(self, X):
        return self.tree.query(X, k=self.k, return_distance=True)
    
    def classify(self, X):
        dist, indices = self.query(X)
        classes = self.train_Y[indices]
        
        predictions = []
        
        for d_s, c_s in zip(dist, classes):
            bins = [0,0]
            for d, c in zip(d_s, c_s):
                bins[c]+=self.weights[c]/(d+1e-5)
            
            #cl = np.asarray([(self.weights[c]/d, c) for d, c in zip(d_s, c_s)])
            
            #print(cl)
            #predictions.append( cl[np.argmax(cl[:,0])][1] )
            predictions.append(np.argmax(bins))
        return predictions


# MAIN PROGRAM
knn_auprc_tot = 0
knn_os_auprc_tot = 0

train_X, train_Y, test_X, test_Y = get_dataset(
        sample=undersampling,
        # Not training the encoder on any outliers gives the best results
        pollution=pollution,  # How much of the outliers to put in the training set
        train_size=train_size  # How much of the inliers to put in the training set
        )

pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components)
        ).fit(train_X)

train_X = pd.DataFrame(pipeline.transform(train_X))

# Drop duplicates after having transformed the training data.
cleaned = pd.concat([train_X,pd.DataFrame(train_Y)],axis=1).drop_duplicates()
train_X = cleaned.iloc[:,:-1]
train_Y = cleaned.iloc[:,-1]

test_X = np.asarray(pipeline.transform(test_X))

knn = KNN(
        k, 
        train_X, 
        train_Y, 
        n_components, 
        weights=weights,
        threshold=threshold
        )

print("predicting outliers based on knn classes")
knn_pred_Y = knn.classify(test_X)

print("predicting outliers based on knn outliers scores")
test_knn_outlier_scores = knn.outlier_score(test_X)

knn_os_pred_Y = [1 if score > knn.threshold else 0 for score in test_knn_outlier_scores]

knn_auprc = average_precision_score(test_Y, knn_pred_Y)

baseline = sum(test_Y)/len(test_Y)

print(confusion_matrix(test_Y, knn_pred_Y))
print(classification_report(test_Y, knn_pred_Y))
print(f"AU-PRC: {knn_auprc}")
print(f"baseline: {baseline}")
knn_auprc_tot += knn_auprc

knn_os_scorer = OutlierDetectorScorer(test_Y, test_knn_outlier_scores)
print(confusion_matrix(test_Y, knn_os_pred_Y))
print(classification_report(test_Y, knn_os_pred_Y))
knn_os_auprc_tot += knn_os_scorer.auprc
print(f"AU-PRC: {knn_os_scorer.auprc}")
print(f"baseline: {baseline}")
print(f"Threshold: {knn.threshold}")
print(f"Optimal threshold: {knn_os_scorer.optimal_thresholds()[0]}")
prc_plot(knn_os_scorer.precisions, knn_os_scorer.recalls, knn_os_scorer.optimal_indices)
plot_report(knn.train_scores, *split_inliers_outliers(test_knn_outlier_scores, test_Y), knn.p, knn.threshold, xscale="log", title="KNN outlier scores")

#print(knn_os_auprc_tot/iterations)
#print(knn_auprc_tot/iterations)

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