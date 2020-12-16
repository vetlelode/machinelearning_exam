import pandas as pd
from preprocessing import get_dataset
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler





"""
from sklearn.decomposition import PCA
pca = PCA(2)
x_pca = pca.fit_transform(X_train)
x_pca = pd.DataFrame(x_pca)
x_pca.columns=['PC1','PC2']

# Plot
import matplotlib.pyplot as plt
plt.scatter(x_pca["PC1"], x_pca["PC2"], c=y_train, alpha=0.8)
plt.title('Scatter plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

clf = AutoEncoder()
"""


def diffs(arrs1, arrs2):
    diff_list = []
    for arr1, arr2 in zip(arrs1, arrs2):
        arr = []
        for a, b in zip(arr1, arr2):
            arr.append(a-b)
        diff_list.append(arr)
    return diff_list

def classify(train_X, train_Y, test_X, test_Y):
    hidden_layers = [10,3,10]
    autoencoder = MLPRegressor(solver="adam", activation="logistic", hidden_layer_sizes = hidden_layers, warm_start=True)
    autoencoder.fit(train_X, train_X)
    predictions = autoencoder.predict(test_X)
    return(diffs(test_X,predictions), test_Y)
    
train_X, train_Y, test_X, test_Y = get_dataset(10000,400,0)

hidden_layers = [10,9,10]
autoencoder = MLPRegressor(solver="adam", activation="relu", hidden_layer_sizes = hidden_layers, warm_start=True)
autoencoder.fit(train_X, train_X)

self_predict = autoencoder.predict(train_X)
sdf = diffs(train_X, self_predict)
predictions = autoencoder.predict(test_X)
df = diffs(test_X, predictions)
#df, Y = classify(*dataset)

from sklearn.decomposition import PCA
pca = PCA(3).fit(sdf)
#pd.DataFrame(StandardScaler().fit_transform(x_train))

x_pca = pca.transform(df)
x_pca = pd.DataFrame(x_pca)
x_pca.columns=['PC1','PC2','PC3']

# Plot
import matplotlib.pyplot as plt
plt.scatter(x_pca["PC1"], x_pca["PC2"], c=test_Y, alpha=0.8)
plt.title('Scatter plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()