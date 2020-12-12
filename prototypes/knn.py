import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from data.preprocessing import get_dataset


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

training, test = get_dataset()

fake = pd.read_csv("../data/fake.csv")
real = pd.read_csv("../data/real.csv")

# Split all the actual scams into two
fake_init, fake_final = train_test_split(fake, test_size=0.5)
real_init, real_final = train_test_split(real, test_size=0.8)

# Create two datasets one with all the data and one split with the ratio set above
full = pd.concat([fake_final, real_final])
full = full.sample(frac=1)

combined = pd.concat(
    [fake_init, real_init.sample(n=len(fake_init) * RATIO * 2)])
combined = combined.sample(frac=1)

X = combined.iloc[:, :-1]
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
print("Results running on the modified dataset:")
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
# if sys.argv[2] == "-p":
#plotDescRegion(X_train, Y_train)

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
