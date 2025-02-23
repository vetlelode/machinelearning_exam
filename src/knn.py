import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scaler import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
import sys

"""
This file contains the various methods used by the various KNN model
The methods of this file are all called from comparison.py using various flags
"""


def runKNN(X_train, Y_train, X_test, K=1) -> list:
    """
    Trains and tests a KNN algorithm on the supplied data and returns the predictions.
    """
    # Scale all the output using a standard scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=K)
    classifier.fit(X_train, Y_train)
    # Return the predicted results
    return classifier.predict(X_test)


def knn_NCA(X_train, Y_train, X_test, K=1) -> list:
    """
    Reduce the dimensionalty of the dataset using the NCA method
    This is slower than using PCA or not using anything at all,
    but yields better results for now

    If the dataset sample is too large this takes really long to run
    """
    # Scale all the output using a standard scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Reduce the dimensionalty of the data using NCA
    nca = NeighborhoodComponentsAnalysis(2).fit(X_train, Y_train)
    X_train_nca = nca.transform(X_train)
    X_test_nca = nca.transform(X_test)

    X_train_nca = pd.DataFrame(X_train_nca)
    X_test_nca = pd.DataFrame(X_test_nca)

    # Classify using a KNN classifier
    clf = KNeighborsClassifier(n_neighbors=K, leaf_size=2)
    clf.fit(X_train_nca, Y_train)
    # Return the predicted results
    return clf.predict(X_test_nca)


def knn_PCA(X_train, Y_train, X_test, K=1) -> list:
    """
    Although PCA peforms worse in most cases from our testing it is useful due to being much faster than NCA
    """
    # Scale all the output using a standard scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    pca = PCA(2)
    pca.fit_transform(X_train)
    pca.fit_transform(X_test)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    classifier = KNeighborsClassifier(n_neighbors=K)
    classifier.fit(X_train, Y_train)
    # Return the predicted results
    return classifier.predict(X_test)


def dim_reduc(X_train, Y_train, X_test, Y_test, K=1) -> None:
    """
    Compare PCA, kernel PCA, and NCA dimensionalty reduction.
    Slightly modified version of this code:
    https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_dim_reduction.html
    Only runs if the -dim argument is provided
    KernelPCA and standard PCA give the same results
    While NCA seems to have a slight edge
    """
    X = pd.concat([X_train, X_test])
    Y = Y_train + Y_test
    random_state = 0
    # Reduce dimension to 2 with PCA
    pca = make_pipeline(StandardScaler(),
                        PCA(n_components=2, random_state=random_state))

    # Reduce dimension to 2 with NeighborhoodComponentAnalysis
    nca = make_pipeline(StandardScaler(),
                        NeighborhoodComponentsAnalysis(n_components=2,
                                                       random_state=random_state))
    # Reduce the dimensionalty using Kernel PCA
    kernel_pca = make_pipeline(StandardScaler(),
                               KernelPCA(2, random_state=random_state))

    # Use a nearest neighbor classifier to evaluate the methods
    knn = KNeighborsClassifier(n_neighbors=K)

    # Make a list of the methods to be compared
    dim_reduction_methods = [('PCA', pca), ('NCA', nca),
                             ('KernelPCA', kernel_pca)]

    # plt.figure()
    for i, (name, model) in enumerate(dim_reduction_methods):
        plt.figure()
        # plt.subplot(1, 3, i + 1, aspect=1)
        # Fit the method's model
        model.fit(X_train, Y_train)
        # Fit a nearest neighbor classifier on the embedded training set
        knn.fit(model.transform(X_train), Y_train)
        # Compute the nearest neighbor accuracy on the embedded test set
        acc_knn = knn.score(model.transform(X_test), Y_test)
        print(name, acc_knn)
        # Embed the data set in 2 dimensions using the fitted model
        X_embedded = model.transform(X)
        # Plot the projected points and show the evaluation score
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                    c=Y, s=30, cmap='Set1',)
        plt.title("KNN with {}\np={}".format(name, round(acc_knn, 3)))
        plt.savefig("figs/KNN_{}.png".format(name))

    plt.show()


def knnGridSearch(X_train, Y_train, X_test, Y_test) -> list:
    """
    Used to run a grid search to find the best params for later usage
    Only runs if the param -grid is provided 
    """
    # Params used for the gird search
    grid_params = {
        'n_neighbors': [1, 3, 5],
    }
    # Scale all the output using a standard scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Reduce the dimensionalty of the data using NCA
    nca = NeighborhoodComponentsAnalysis(2).fit(X_train, Y_train)
    X_train_nca = nca.transform(X_train)
    X_test_nca = nca.transform(X_test)
    # Run the Grid search and print out the best params
    classifier = KNeighborsClassifier()
    gs = GridSearchCV(classifier, grid_params, verbose=1, cv=3, n_jobs=-1)
    gs.fit(X_train_nca, Y_train)
    print(gs.best_params_)
    # Score the best found params using a confusion matrix
    Y_pred = gs.predict(X_test_nca)
    print(confusion_matrix(Y_test, Y_pred))
