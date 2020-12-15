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
        for a, b in zip(arr1, arr2):
            diff_list.append(a-b)
    return diff_list

def classify(train_X, train_Y, test_X, test_Y):
    print(train_X)
    hidden_layers = [20,10,5,10,20]
    autoencoder = MLPRegressor(solver="adam", activation="logistic", hidden_layer_sizes = hidden_layers, warm_start=True)
    autoencoder.fit(train_X, train_X)
    predictions = autoencoder.predict(test_X)
    print(predictions, len(predictions))
    #print(diffs(test_X,predictions))
dataset = get_dataset(400,400,0)
print(classify(*dataset))