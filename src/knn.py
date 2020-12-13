import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import sys
from plotting import plotConfusion

"""
Prototype implmentation of an KNN based solution with a 50/50 dataset
"""


def runKNN(X_train, Y_train, X_test, Y_test) -> list:
    """
    Trains and tests a KNN algorithm on the supplied data and returns the predictions.
    """
    print(len(X_train), len(X_test))

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    return Y_pred


def knnGridSearch(X_train, Y_train, X_test, Y_test) -> list:
    """
    Used to run a grid search to find the best params for later usage
    """
    grid_params = {
        'n_neighbors': [1, 3, 5],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski', 'manhattan'],
    }
    reduced_params = {
        'n_neighbors': [1],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski', 'manhattan'],
    }
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = KNeighborsClassifier()
    gs = GridSearchCV(classifier, grid_params, verbose=1, cv=3, n_jobs=-1)
    res = gs.fit(X_train, Y_train)
    print(res)
    pred = gs.predict(X_test)
    print(confusion_matrix(Y_test, pred))
