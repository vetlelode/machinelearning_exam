from sklearn.metrics import classification_report, confusion_matrix
import sys
import numpy as np
import pandas as pd
from preprocessing import get_dataset
from knn import runKNN, knnGridSearch, knn_NCA, dim_reduc
import seaborn as sn
import matplotlib.pyplot as plt


def runComp():
    X_train, Y_train, X_test, Y_test = get_dataset(5000, 200, 0.5)
    cols = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15",
            "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]
    X_train = pd.DataFrame(X_train, columns=cols)
    X_test = pd.DataFrame(X_test, columns=cols)

    Y_pred = runKNN(X_train, Y_train, X_test, Y_test)
    cf_knn = confusion_matrix(Y_test, Y_pred)
    print("Confusion matrix for KNN:\n{}".format(cf_knn))
    print("False positive rate: {}%".format(
        cf_knn[1][0]/len(Y_test)))
    if len(sys.argv) >= 2:
        if sys.argv[1] == "-grid":
            findBestParams(X_train, Y_train, X_test, Y_test)
        elif sys.argv[1] == "-corr":
            # Print out a correlation matrix for the entire dataset, allowing some limited insight into the correlation of the attributes to the class
            correlationMatrix()
        elif sys.argv[1] == "-nca":
            # Run the KNN classifier, but with NCA dimensionalty reduction, which from our testing gives slightly better results than PCA
            Y_pred_nca = knn_NCA(X_train, Y_train, X_test, Y_test, 1)
            cf_knn_nca = confusion_matrix(Y_test, Y_pred_nca)
            print("Confusion matrix for KNN with NCA:\n{}".format(cf_knn_nca))
            print("False positive rate: {}".format(
                cf_knn_nca[1][0]/len(Y_test)))
        elif sys.argv[1] == "-dim":
            # Function comparing the results of PCA and NCA
            dim_reduc(X_train, Y_train, X_test, Y_test, 1)


def correlationMatrix():
    """
    Create a correlation matrix using the methodology form here:
    https://datatofish.com/correlation-matrix-pandas/
    """
    df = pd.read_csv("../data/creditcard.csv")
    corrMatrix = df.corr()
    res = corrMatrix.sort_values(by=['Class'], ascending=False)
    print(res['Class'])
    sn.heatmap(corrMatrix, annot=True)
    plt.show()


def findBestParams(X_train, Y_train, X_test, Y_test):
    knnGridSearch(X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
    runComp()
