import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

"""
Prototype implmentation of an KNN based solution with a 50/50 dataset
"""

fake = pd.read_csv("../data/fake.csv")
real = pd.read_csv("../data/real.csv")
real = real.sample(n=492)
combined = pd.concat([fake, real])
combined = combined.sample(frac=1)
X = combined.iloc[:, :-1]
print(X)
Y = combined['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier()
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

# Print out various metrics about the accuracy on a 50/50 dataset
print(classifier.score(X_test, Y_test))
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


# Only run on the full dataset if -f is provided as a argument
if len(sys.argv) >= 2 and sys.argv[1] == "-f":
    print("Running the KNN classifier on the full dataset:")
    full = pd.read_csv("../data/creditcard.csv")
    print(combined.shape)
    print(full.shape)
    X_test_full = full.iloc[:, :-1]
    print(X_test_full)
    X_test_full = scaler.transform(X_test_full)
    Y_test_full = full['Class']
    print(full)
    Y_pred_full = classifier.predict(X_test_full)

    print(classification_report(Y_test_full, Y_pred_full))
