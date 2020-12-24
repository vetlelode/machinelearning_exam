from sklearn.metrics import classification_report, confusion_matrix
import sys
import numpy as np
import pandas as pd
from preprocessing import get_dataset
from knn import runKNN, knnGridSearch, knn_NCA, dim_reduc
import seaborn as sn
import matplotlib.pyplot as plt

"""
Compare various methods and paramaters for running the KNN classifier
With the two main ones being standard KNN and KNN with PCA dimensionalty reduction.
KNN with PCA generally has a slight edge in peformance.
"""


def runComp():
    """
    Run the KNN algorithm on the dataset with the provided flags
    """
    k=3
    X_train, Y_train, X_test, Y_test = get_dataset(sample=50000,pollution=0.7,train_size=0.8)
    cols = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15",
            "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]
    X_train = pd.DataFrame(X_train, columns=cols)
    X_test = pd.DataFrame(X_test, columns=cols)
    # X_train.drop(columns="Time")
    # X_test.drop(columns="Time")
    Y_pred = runKNN(X_train, Y_train, X_test, k)
    # Print out the confusion matrix since its more relevant than the overall accuracy
    cf_knn = confusion_matrix(Y_test, Y_pred)
    print("Confusion matrix for KNN:\n{}".format(cf_knn))
    print("Classification report for standard KNN:\n {}".format(
        classification_report(Y_test, Y_pred)))
    if len(sys.argv) >= 2:
        if "-grid" in sys.argv:
            # Run a grid search to find the overall best configuration for the KNN classifier.
            knnGridSearch(X_train, Y_train, X_test, Y_test)
        elif "-corr" in sys.argv:
            # Print out a correlation matrix for the entire dataset, allowing some limited insight into the correlation of the attributes to the class
            correlationMatrix()
        elif "-nca" in sys.argv:
            # Run the KNN classifier, but with NCA dimensionalty reduction, which from our testing gives slightly better results than PCA
            Y_pred_nca = knn_NCA(X_train, Y_train, X_test, k)
            # Print out the confusion matrix since its more relevant than the overall accuracy
            cf_knn_nca = confusion_matrix(Y_test, Y_pred_nca)
            print("Confusion matrix for KNN with NCA:\n{}".format(cf_knn_nca))
            print("Classification report for KNN with NCA:\n {}".format(
                classification_report(Y_test, Y_pred_nca)))
        elif "-dim" in sys.argv:
            # Function comparing the results of PCA and NCA
            dim_reduc(X_train, Y_train, X_test, Y_test, k)


def correlationMatrix():
    """
    Create a correlation matrix using the methodology form here:
    https://datatofish.com/correlation-matrix-pandas/
    This isn't that useful, but is interesting for finding relevant correlations 
    """
    df = pd.read_csv("../data/creditcard.csv")
    corrMatrix = df.corr()
    res = corrMatrix.sort_values(by=['Class'], ascending=False)
    print(res['Class'])
    sn.heatmap(corrMatrix, annot=True)
    plt.show()


if __name__ == "__main__":
    runComp()
