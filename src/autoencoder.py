import pandas as pd
import numpy as np
from preprocessing import get_dataset
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline



#Hyperparameters:
hidden_layers = [20,10,2,10,20]
components = 28
score_threshold = -50

class AutoEncoderErrorPCA:
    def __init__(
            self, 
            solver       : str   ="adam", 
            activation   : str   = "relu", 
            hidden_layer_sizes : list = (20,20,5,20,20), 
            warm_start   : bool  = False,
            learning_rate: float = "adaptive",
            max_iter     : int   = 200,
            tol          : float = 1e-6,
            classifier   : str   = "PCA",
            components   : int   = None,
            threshold    : float = None
            ):
        
        clf = ("PCA", "difmean","errorlength")
        if classifier not in clf:
            raise Exception("invalid classifier")
        
        self.regressor = MLPRegressor(
            solver=solver,
            activation=activation,
            hidden_layer_sizes = hidden_layer_sizes,
            warm_start=warm_start,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol
        )
        
        self.classifier = classifier
        self.components = components
        self.threshold  = threshold


#Diagrams - Details unimportant
def plot_histogram(X, Y, title) -> None:
    s = sorted(X)
    lower_bound, upper_bound = s[40], s[-40]
    
    X_inliers = [x for x,y in zip(X,Y) if y==0 ]
    X_outliers = [x for x,y in zip(X,Y) if y==1 ]
    
    bins = np.linspace(lower_bound, upper_bound, 100)
    plt.hist(x=(X_inliers, X_outliers), bins=bins, alpha=0.5, label=('inliers','outliers'), stacked=True, histtype="stepfilled")
    plt.legend(loc='upper right')
    plt.title(title)
    plt.show()


#Diagrams - Details unimportant
def plot_scatter(pca: list, Y) -> None:
    X = pd.DataFrame(pca)
    for i in X:
        plot_histogram(X[i],Y,i)


def predict_score_threshold(scores,threshold):
    return [1 if score < threshold else 0 for score in scores]


def r2_score_samples(autoencoder, X):
    return [r2_score(y_true, y_pred) for y_true, y_pred in zip(outliers, auto_encoder.predict(outliers))]



#def run_autoencoder_outlier_detection(hidden_layers, components, score_threshold):
#Obtain the dataset
train_X, train_Y, test_X, test_Y = get_dataset(k1=80000,f=0)

pipeline = Pipeline([('normalizer', Normalizer()),
                     ('scaler', StandardScaler())])
scaler = StandardScaler()
pipeline = scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)
# Train a neural net into accurately recreate the input data 
# through a small latent space.
auto_encoder = MLPRegressor(
    solver="adam",
    activation="tanh", #Relu works well for this
    hidden_layer_sizes = hidden_layers,
    warm_start=False, #Used in debugging
    #learning_rate="adaptive",
    max_iter=50,
    verbose=True,
    tol=1e-7
    )
# The autoencoder is only being trained on inliers as to not learn to
# recreate outliers explicitly
auto_encoder.fit(train_X, train_X)

# Take the difference between the input vector and the output vector 
# of the auto encoder
training_error = train_X - auto_encoder.predict( train_X )

inliers = [x for x,y in zip(test_X,test_Y) if y==0]
outliers = [x for x,y in zip(test_X,test_Y) if y==1]
recreations = auto_encoder.predict( test_X )
test_error     = test_X - recreations
error_inliers = [x for x,y in zip(test_error,test_Y) if y==0]
error_outliers = [x for x,y in zip(test_error,test_Y) if y==1]


# Perform principal component analysis with respect to the training_error
pca = PCA(components).fit( training_error )

# Score the test errors based on the principal component analysis of the 
# training error. This is the likelihood of the test sample being explained
# by the elipsoid from the PCA
scores = pca.score_samples(test_error)
scores_inliers = [x for x,y in zip(scores,test_Y) if y==0]
scores_outliers = [x for x,y in zip(scores,test_Y) if y==1]

# In some earlier tests, we attempted to measure the length of the error
# vectors to adequate results, but this assumes that the pca elipsoid is a 
# perfect sphere. Some axes have higher variance and are therefore more 
# important


# Predict whether the samples are inliers or outliers based off a threshold
# If a sample has a score less than the threshold, it is unlikely to be
# explained by the PCA elipsoid
predicted_Y = predict_score_threshold(scores,score_threshold)


plot_histogram(scores, test_Y, 'PCA scores of the autoencoder error')
plot_histogram(r2_score_samples(auto_encoder, test_X), test_Y, 'r2 scores')
#plot_scatter(pca.transform(test_X), test_Y)

#plot_scatter(test_pca, test_Y)
#plot_scatter(training_pca, train_Y)
print(confusion_matrix(test_Y, predicted_Y))
print(classification_report(test_Y, predicted_Y))
training_scores = pca.score_samples(training_error)
print("minimum training score:%s"%min(training_scores))
print("max outlier score: %s"%max([a for a,y in zip(scores,test_Y) if y==1]))
print("min inlier score: %s"%min([a for a,y in zip(scores,test_Y) if y==0]))
print("avg training score: %s"%pca.score(training_error))
print("avg inlier score: %s"%pca.score(error_inliers))
print("avg outlier score: %s"%pca.score(error_outliers))
print("std training: %s"% np.std(training_scores))
print("std inliers: %s"% np.std(scores_inliers))
print("std outliers: %s"% np.std(scores_outliers))


plt.matshow(pd.DataFrame(pca.transform(test_error)).corr())
plt.show()

