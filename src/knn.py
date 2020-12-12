import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import sys
from preprocessing import get_dataset


def plotDescRegion(X, y) -> None:
    from mlxtend.plotting import plot_decision_regions
    print(y.to_numpy())
    plot_decision_regions(X[:, [0, 1]], y.to_numpy(), clf=classifier)
    plt.show()


"""
Prototype implmentation of an KNN based solution with a 50/50 dataset
"""
# Change this to change the ratio of real to fake in the training set
RATIO = 20

X_train, Y_train, X_test, Y_test = get_dataset(20000, 450)
cols = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15",
        "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "Class"]
X_train = pd.DataFrame(X_train, columns=cols)
Y_train = pd.DataFrame(Y_train, columns=cols)
X_test = pd.DataFrame(X_test, columns=cols)
Y_test = pd.DataFrame(Y_test, columns=cols)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier()
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

# Print out various metrics about the accuracy on a 50/50 dataset
print("Results running on the modified dataset:")
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
# if sys.argv[2] == "-p":
# plotDescRegion(X_train, Y_train)

# Only run on the full dataset if -f is provided as a argument
if len(sys.argv) >= 2 and sys.argv[1] == "-f":
    print("Running the KNN classifier on the full dataset:")

    X_test_full = full.iloc[:, :-1]
    X_test_full = scaler.transform(X_test_full)
    Y_test_full = full['Class']
    Y_pred_full = classifier.predict(X_test_full)
    # Print out results of the full runthrough
    print(confusion_matrix(Y_test_full, Y_pred_full))
    print(classification_report(Y_test_full, Y_pred_full))
    # Only graph if argument -p is given
    # if sys.argv[2] == "-p":
    # plot_decision_regions(X_train, Y_train, clf=classifier)
    # plt.show()
