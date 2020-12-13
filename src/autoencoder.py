from preprocessing import get_dataset
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

train_X, train_Y, test_X, test_Y = get_dataset(2000,400,0)
hidden_layers = [100,20,3,20,100]
autoencoder = MLPRegressor(solver="adam", hidden_layer_sizes = hidden_layers, warm_start=True)
autoencoder.fit(train_X, train_X)

autoencoder_predictions = autoencoder.predict(train_X)

n = len(test_X)
clf_X, clf_Y, t_X, t_Y = test_X[:n//2], test_Y[:n//2], test_X[n//2:], test_Y[n//2:]

clf = MLPClassifier(activation = "logistic", hidden_layer_sizes=(20,10))
clf.fit(clf_X, clf_Y)
predictions = clf.predict(t_X)
print(confusion_matrix(predictions, t_Y))
