from sklearn.metrics import classification_report, confusion_matrix
import sys
import numpy as np
import pandas as pd
from preprocessing import get_dataset
from knn import runKNN, knnGridSearch


def runComp():
    X_train, Y_train, X_test, Y_test = get_dataset(50000, 450, 0.2)
    cols = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15",
            "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]
    X_train = pd.DataFrame(X_train, columns=cols)
    # Y_train = pd.DataFrame(Y_train, columns=["Class"])
    X_test = pd.DataFrame(X_test, columns=cols)

    # Y_test = pd.DataFrame(Y_test, columns=["Class"])
    # Run a grid search to find the best params if the argument -grid is provdided
    if len(sys.argv) >= 2 and sys.argv[1] == "-grid":
        findBestParams(X_train, Y_train, X_test, Y_test)
    knn = runKNN(X_train, Y_train, X_test, Y_test)
    cf_knn = confusion_matrix(Y_test, knn)
    print("Confusion matrix for KNN:\n{}".format(cf_knn))


def findBestParams(X_train, Y_train, X_test, Y_test):
    knnGridSearch(X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
    runComp()
