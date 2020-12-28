import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from preprocessing import get_dataset


def StandardScaler(X_train, X_test):
    """
    Scale the two provided datasets
    Uses Numpy for peformance and simplicity
    """
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    # Run the scaler for all collumns
    scaler_vals = np.empty((30, 2))
    for i in range(len(X_train[0])):
        column = X_train[:, i]
        test_col = X_test[:, i]
        # U
        U = np.mean(column)
        # S
        S = np.std(column)
        scaler_vals[i, 0] = U
        scaler_vals[i, 1] = S
    # Scale all values of X_train
    for X in range(len(X_train)):
        for y in range(len(X_train[0])):
            X_train[X, y] = (X_train[X, y]-scaler_vals[y, 0])/scaler_vals[y, 1]
    # Scale all values of X_test
    for X in range(len(X_test)):
        for y in range(len(X_test[0])):
            X_test[X, y] = (X_test[X, y]-scaler_vals[y, 0])/scaler_vals[y, 1]


class KNN:
    def __init__(self, X_train, Y_train, X_test, K=3):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.K = K

    def scale(self):
        """
        Scale all the provided data
        """
        print(self.X_train, self.Y_train)

    def get_neighbours(self):
        distances = list()
        for row in self.X_train:
            dist = self.euclidean_distance(row, self.X_train[0])
            distances.append((row, dist))

    def euclidean_distance(self, row1, row2):
        """
        Calculate the euclidean distance between Row1 and Row2
        """
        dist = 0.0
        for i in range(len(row1) - 1):
            dist += (row1[i] - row2[i])**2
        return math.sqrt(dist)


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_dataset(
        sample=1000, pollution=0.7, train_size=0.9)
    knn = KNN(X_train, Y_train, X_test, K=3)
    knn.get_neighbours()
    StandardScaler(X_train, X_test)
