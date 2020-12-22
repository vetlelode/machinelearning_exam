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

from scipy.stats import invgamma



#Hyperparameters:
hidden_layers = [20,10,2,10,20]
components = 28
activation = "tanh"
threshold = 0.9

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
def plot_histogram(X, Y, title, threshold=None) -> None:
    #Omit extreme outliers
    std = np.std(X)
    mean = np.mean(X)
    lower_bound, upper_bound = max(min(X), mean-2*std), min(max(X),mean+2*std)
    
    X_inliers = [x for x,y in zip(X,Y) if y==0 ]
    X_outliers = [x for x,y in zip(X,Y) if y==1 ]
    
    bins = np.linspace(lower_bound, upper_bound, 150)
    
    fig, ax = plt.subplots()
    ax.hist(x=(X_inliers, X_outliers), bins=bins, alpha=0.5, label=('inliers','outliers'), stacked=False, histtype="stepfilled", density=True)
    ax.legend(loc='upper left')
    if threshold != None:
        min_ylim, max_ylim = plt.ylim()
        ax.vlines(threshold,min_ylim,max_ylim)
    plt.title(title)
    plt.show()


#Diagrams - Details unimportant
def plot_scatter(X,Y,title,threshold=None) -> None:
    fig, ax = plt.subplots()
    scatter = ax.scatter(range(len(X)),X,c=Y, alpha=0.3)
    handles, labels = scatter.legend_elements()
    ax.legend(handles, ["inliers", "outliers"], loc='lower left')
    if threshold != None:
        min_xlim, max_xlim = plt.xlim()
        ax.hlines(threshold,min_xlim,max_xlim)
    plt.title(title)
    plt.show()



def threshold_predict(scores,threshold):
    return [1 if score < threshold else 0 for score in scores]

#https://i-systems.github.io/teaching/ML/iNotes/15_Autoencoder.html
def encode(reg, X, layers:int, activation=activation):
    data = np.asmatrix(X)
    coefficients = reg.coefs_
    intercepts = reg.intercepts_
    activations = {
            "tanh": lambda a: (np.exp(a) - np.exp(-a))/(np.exp(a) + np.exp(-a)),
            "relu": np.vectorize(lambda a: a if a>0 else 0)
            }
    activation_function = activations[activation]
    layer = activation_function(data*coefficients[0] + intercepts[0])
    for i in range(1,layers):
        layer = activation_function(layer*coefficients[i] + intercepts[i])
    return layer


class LogLikelihood:
    def __init__(self, X):
        X = np.asmatrix(X)
        self.n_axes = X.shape[1]
        # log likelihood parameters
        # the mean of each axis
        self.μs = [np.mean(X[:,i]) for i in range(self.n_axes)]
        # the standard deviation of each axis
        self.σ2s = [np.var(X[:,i]) for i in range(self.n_axes)]
    
    def score(self, Y):
        Y = np.asmatrix(Y)
        # The PDF of the normal distribution is
        # e^(-(x - μ)^2/(2 σ^2))/(sqrt(2 π) σ)
        # Which is isomorphic to
        # -(x - μ)^2 / (σ^2)
        # the exact probabilities are not important to the scoring,
        # only that two scores in one distribution have the same
        # relationship as in the other (f(a)<f(b) <=> g(a)<g(b))
        
        # We can then sum the log-likelihood of each axis
        # as that is isomorphic to taking the product of the likelihood
        
        log_likelihoods = []
        for y in Y:
            ll = sum([(y[0,i]-self.μs[i])**2 / self.σ2s[i] for i in range(self.n_axes)])
            log_likelihoods.append( ll )
        return log_likelihoods

def predict(scoring_function, x_train, x_test, std_factor=1):
    # Score the training data. These should have a low mean score
    training_scores = scoring_function( x_train )
    # Set the threshold to be std_factor standard deviations below the mean
    # score of the training data
    threshold = np.mean(training_scores)-std_factor*np.std(training_scores)
    
    # Score the test data...
    test_scores = scoring_function( x_test )
    # Predict test class based off of whether their score is below the 
    # threshold
    return test_scores, threshold_predict( test_scores, threshold ), threshold


def plot_scores(scores, Y, title, threshold=None):
    plot_histogram( scores, Y, title, threshold)
    plot_scatter  ( scores, Y, title, threshold)


# Obtain the dataset
train_X, train_Y, test_X, test_Y = get_dataset(f=0)

# Scale data to make training easier
# -
# We tried to normalize and scale, but that made it *too*
# easy to train, and we got extreme overfitting and the model
# would fit new inliers and outliers equaly poorly.
# A way to avoid overfitting would be to introduce noise to the training data
# per epoch, but sklearn does not support this.
# Standardscaler works fairly well on its own.
scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)


# Train a neural net into accurately recreate the input data 
# through a small latent space.
auto_encoder = MLPRegressor(
    solver="adam",
    activation=activation, #Relu also works well, but easily leads to dead neurons
    hidden_layer_sizes = hidden_layers,
    warm_start=False, #Used in debugging
    max_iter=50,
    verbose=True,
    tol=1e-7
    )


# The autoencoder is only being trained on inliers as to not learn to
# recreate outliers explicitly
auto_encoder.fit(train_X, train_X)


# Plot out the latent space
latent = np.asarray(encode(auto_encoder, test_X, 3, activation))
latent_inliers = np.asmatrix([x for x,y in zip(latent,test_Y) if y==0])
latent_outliers = np.asmatrix([x for x,y in zip(latent,test_Y) if y==1])
plt.figure(figsize = (10,10))
plt.scatter(np.asarray(latent_inliers[:,0]),np.asarray(latent_inliers[:,1]), label = '0', alpha=0.3)
plt.scatter(np.asarray(latent_outliers[:,0]),np.asarray(latent_outliers[:,1]), label = '1', alpha=0.3)
plt.title('Latent Space', fontsize=15)
plt.xlabel('Z1', fontsize=10)
plt.ylabel('Z2', fontsize=10)
plt.axis('equal')
plt.show()

# Recreate the data from the auto encoder
train_recreations = auto_encoder.predict( train_X )
test_recreations  = auto_encoder.predict( test_X  )

# Score the samples with r2. This is the loss function used in the autoencoder.
# This is done because sklearn auto encoder does not support per sample scoring
train_r2_scores = [r2_score( x_true, x_pred ) for x_true, x_pred in zip(train_X, train_recreations)]
test_r2_scores  = [r2_score( x_true, x_pred ) for x_true, x_pred in zip(test_X, test_recreations)]


# Find a decent threshold to split inliers and outliers
# Use the 68% rule (1 standard deviation)
r2_threshold = np.mean(train_r2_scores) - np.std(train_r2_scores)

# Predict the test data wrt the r2 threshold
r2_predictions = threshold_predict(test_r2_scores, r2_threshold)


# Plot the r2 scores
plot_scores(test_r2_scores, test_Y, 'R2 scores',r2_threshold)


# R2 scoring does not take into account that the different axes
# might have different variances. Since the model is trained
# only on inliers, it might happen that the distinguishing
# axes of the outliers have low variance, and therefore R2 will
# not pick up on that.

# Take the error between the original data and its recreation
training_errors = train_X - train_recreations
test_errors     = test_X  - test_recreations

LL = LogLikelihood(training_errors)

# Score the training data. These should have a low mean score
training_ll_scores = LL.score(training_errors)
test_ll_scores     = LL.score(test_errors)

a, loc, scale = invgamma.fit(training_ll_scores)
ll_thresh = invgamma.isf(1-threshold, a, loc, scale)


bins = np.linspace(0, 200, 150)
fig, ax = plt.subplots()
ax.hist(training_ll_scores, bins)
min_ylim, max_ylim = plt.ylim()
ax.plot(bins, np.multiply(invgamma.pdf(bins, a, loc, scale),max_ylim*20))

plt.title("log likelihood training scores")
plt.show()

# The log likelihood is the sum of logs, which the expected error follows
# the gamma distribution, where the expected error is digamma(k)+ln(theta)
# or digamma(alpha) - ln(beta)

#variance/mean = theta
#mean^2 /variance=kk

ll_predictions = [1 if x > ll_thresh else 0 for x in test_ll_scores]


# Plot the ll scores
plot_scores(
        np.log(np.abs(test_ll_scores)), 
        test_Y, 
        'log(abs(log likelihood scores))', 
        np.log(np.abs(ll_thresh))
        )



# Another method is to take a principal component analysis of the data
# This takes into account that some axes are more important than others.
# As the autoencoder scores are subject to the curse of dimensionality


# Fit the pca to the training errors.
# We have tried a number of different amount of components
# but considering the axes are picked based off of the variance of the training 
# data, omiting the lower variance axes might also omit the characteristic axes
# of the outliers. We include all axes of the PCA.
# This is only to transform the errors to an ellipsoid space
# where if the datapoints are within the ellipsoid, the point is most likely
# an inlier.
# Points outside the elipsoid on a minor axis will have a major impact on its
# likelihood of being an outlier
aepca = PCA(components).fit( training_errors )

test_aepca_scores, aepca_predictions, aepca_thresh = predict(
        aepca.score_samples, 
        training_errors, 
        test_errors, 
        0.3
        )

# Plot the AE-PCA scores
plot_scores(
        np.log(np.abs(test_aepca_scores)),
        test_Y, 
        'ln(abs(AE-PCA scores))',
        np.log(np.abs(aepca_thresh))
        )

# It is worth noting that PCA performs many of the same tasks
# as a simple autoencoder would do, and we would get similar results
# by feeding the training data directly into the pca

direct_pca = PCA(components).fit( train_X )
test_dpca_scores, dpca_predictions, dpca_thresh = predict(
        direct_pca.score_samples, 
        train_X, 
        test_X, 
        0.3
        )


# Plot the PCA scores
plot_scores(
        np.log(np.abs(test_dpca_scores)),
        test_Y, 
        'ln(abs(Direct-PCA scores))',
        np.log(np.abs(dpca_thresh))
        )



# It seems r2 scoring performs better when we sample more data
print("r2 report:")
print(confusion_matrix(test_Y, r2_predictions))
print(classification_report(test_Y, r2_predictions))

# But LL has stable performance, even when undersampling
print("LL report:")
print(confusion_matrix(test_Y, ll_predictions))
print(classification_report(test_Y, ll_predictions))

print("AE-PCA report:")
print(confusion_matrix(test_Y, aepca_predictions))
print(classification_report(test_Y, aepca_predictions))

print("Direct-PCA report:")
print(confusion_matrix(test_Y, dpca_predictions))
print(classification_report(test_Y, dpca_predictions))


inlier_errors = [x for x,y in zip(test_errors,test_Y) if y==0]
outlier_errors = [x for x,y in zip(test_errors,test_Y) if y==1]

plt.matshow(pd.DataFrame(inlier_errors).corr())
plt.title("Correlation matrix: inlier errors")
plt.show()

plt.matshow(pd.DataFrame(outlier_errors).corr())
plt.title("Correlation matrix: outlier errors")
plt.show()

plt.matshow(pd.DataFrame(training_errors).corr())
plt.title("Correlation matrix: training errors")
plt.show()