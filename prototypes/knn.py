import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Simple prototype of KNN on the dataset

#data_raw = pd.read_csv("../data/creditcard.csv")
fake = pd.read_csv("../data/fake.csv")
real = pd.read_csv("../data/real.csv")

real = real.sample(n=492)
combined = pd.concat([fake, real])
combined = combined.sample(frac=1)
print(combined)
X = combined.iloc[:, :-1].values
Y = combined['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
print(Y_train.shape)
#scaler = StandardScaler()
# scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier()
classifier.fit(X_train, Y_train)
print(classifier.score(X_test, Y_test))
Y_pred = classifier.predict(X_test)
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
