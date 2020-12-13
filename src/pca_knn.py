# -*- coding: utf-8 -*-
from preprocessing import get_dataset
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix



x_train, y_train, x_test, y_test = get_dataset(400,400)

x_train = pd.DataFrame(StandardScaler().fit_transform(x_train))
x_test = pd.DataFrame(StandardScaler().fit_transform(x_test))

pca = PCA(2).fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

x_train_pca = pd.DataFrame(x_train_pca)
x_test_pca = pd.DataFrame(x_test_pca)


plt.scatter(x_train_pca[0], x_train_pca[1], c=y_train, alpha=0.8)
plt.title('Scatter plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


clf = KNeighborsClassifier(n_neighbors=7, weights="distance")
print(clf)
clf.fit(x_train_pca, y_train)
y_pred = clf.predict(x_test_pca)

# Print out various metrics about the accuracy on a 50/50 dataset
print("Results running on the modified dataset:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))