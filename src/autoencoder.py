import pandas as pd
import numpy as np
from preprocessing import get_dataset
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

#Hyperparameters:
hidden_layers = [10,5,10]
n_of_pca = 5
distance_threshold = 10
score_threshold = -50


def plot_histogram(inliers, outliers, title):
    lower_bound = min(min(inliers),min(outliers))
    upper_bound = max(max(inliers),max(outliers))
    bins = np.linspace(lower_bound, upper_bound, 100)
    plt.hist(inliers, bins, alpha=0.5, label='inliers')
    plt.hist(outliers, bins, alpha=0.5, label='outliers')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.show()


def plot_scatter(pca, Y):
    X = pd.DataFrame(pca)
    for i in X:
        for j in X:
            if i==j: continue
            plt.scatter(X[i], X[j], c=Y, alpha=0.8)
            plt.title('Relationships between principal components')
            plt.xlabel('pc%s' % i)
            plt.ylabel('pc%s' % j)
            plt.show()

def predict_distance_threshold(distances):
    return [1 if d > distance_threshold else 0 for d in distances]

def predict_score_threshold(scores):
    return [1 if score < score_threshold else 0 for score in scores]


#Obtain the dataset
train_X, train_Y, test_X, test_Y = get_dataset(100000,400,0)


auto_encoder = MLPRegressor(
    solver="adam",
    activation="relu", #Relu works well for this
    hidden_layer_sizes = hidden_layers,
    warm_start=False #Used in debugging
    )
auto_encoder.fit(train_X, train_X)

#take the difference between the input vector and the output vector of the auto encoder
training_error = train_X - auto_encoder.predict( train_X )
test_error     = test_X  - auto_encoder.predict( test_X  )


#Perform principal component analysis with respect to the training_error

pca = PCA(n_of_pca).fit( training_error )
#transform the test error to this space
#this is done because the analysis is useless with few samples,
#and we want to be able to reliably use the auto encoder to predict single instances
test_pca = pca.transform( test_error )
training_pca = pca.transform( training_error )

#Take the absolute distance of the errors after pca
distances = [ sum(row[:]**2)**0.5 for row in test_pca ]
distances_inliers = [d for d,y in zip(distances,test_Y) if y == 0]
distances_outliers = [d for d,y in zip(distances,test_Y) if y == 1]

scores = pca.score_samples(test_error)
scores_inliers = [err for err, y in zip(scores, test_Y) if y==0]
scores_outliers =[err for err, y in zip(scores, test_Y) if y==1]

plot_histogram(distances_inliers, distances_outliers, 'vector length of distance between input and output')

plot_histogram(scores_inliers, scores_outliers, 'PCA scores of distance between input and output')
pred_Y_d = predict_distance_threshold(distances)
pred_Y_s = predict_score_threshold(scores)
#plot_scatter(test_pca, test_Y)
#plot_scatter(training_pca, train_Y)
print(confusion_matrix(test_Y, pred_Y_d))

print(confusion_matrix(test_Y, pred_Y_s))