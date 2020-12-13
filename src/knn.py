import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import sys
from plotting import plotConfusion

"""
Prototype implmentation of an KNN based solution with a 50/50 dataset
"""


def runKNN(X_train, Y_train, X_test, Y_test) -> list:

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    return [Y_test, Y_pred]
