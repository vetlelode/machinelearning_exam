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
# Higher threshold means less false positives, but also less true negatives
threshold = 0.9



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


def encode(network, X):
    data = np.asmatrix(X)
    z = data
    for weight, bias, activation in network:
        # concatenate the bias to fit the weight-layer multiplication
        matbias = [bias]*z.shape[0]
        # sum the weights and then add the bias
        z = activation(np.matmul(z, weight) + matbias)
    return np.asarray(z)


class LogLikelihood:
    def __init__(self, X):
        X = np.asmatrix(X)
        self.X = X
        self.n_axes = X.shape[1]
        # Create a n-dimensional normal distribution:
        
        # log likelihood parameters
        # the mean of each axis
        self.μs = [np.mean(X[:,i]) for i in range(self.n_axes)]
        # the standard deviation of each axis
        self.σ2s = [np.var(X[:,i]) for i in range(self.n_axes)]
    
    def score(self, Y):
        Y = np.asarray(Y)
        # The PDF of the normal distribution is
        # e^(-(x - μ)^2/(2 σ^2))/(sqrt(2 π) σ)
        # Which is isomorphic to
        # (x - μ)^2 / (σ^2)
        # the exact probabilities are not important to the scoring,
        # only that two scores in one distribution have the same
        # relationship as in the other R(f(a),f(b)) <=> R(g(a),g(b))
        
        # f(x)=e^(-(x - μ)^2/(2 σ^2))/(sqrt(2 π) σ)
        # g(x)=(x - μ)^2 / (σ^2)
        
        # ln(f(x))=-(x - μ)^2/(2 σ^2)-ln(sqrt(2 π) σ)
        # ln(f(x))+ln(sqrt(2 π) σ)=-(x - μ)^2/(2 σ^2)
        # -2(ln(f(x))+ln(sqrt(2 π) σ)) = (x - μ)^2 / (σ^2)
        # g(x) = -ln(f(x)^2 2 π σ^2)
        
        
        # We can then sum the log-likelihood of each axis
        # as that is isomorphic to taking the product of the likelihood
        return sum([(Y.T[i]-self.μs[i])**2 / self.σ2s[i] for i in range(self.n_axes)])
    
    
    def gamma_threshold(self, threshold):
        # The density of values of the scores of the training data
        # follows the gamma distribution.
        # A sharp spike of likelihood followed by a sharp decline that 
        # smoothens out slowly.
        
        # In other words, it does not follow a normal distribution,
        # and as such, a more sohpisticated method to find the apropriate
        # threshold is warranted.
        
        # Calculate the describing parameters of the training data 
        # gamma distribution, and the threshold
        a, loc, scale = invgamma.fit(self.score(self.X))
        return invgamma.isf(1-threshold, a, loc, scale), (a, loc, scale)


def plot_scores(scores, Y, title, threshold=None):
    plot_histogram( scores, Y, title, threshold)
    plot_scatter  ( scores, Y, title, threshold)




# Obtain the dataset
train_X, train_Y, test_X, test_Y = get_dataset(k1=2000,f=0)

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
n_layers = len(auto_encoder.coefs_)
activations = [np.tanh]*(n_layers-1)+[lambda a:a]
network = list(zip(auto_encoder.coefs_, auto_encoder.intercepts_, activations))

latent = encode(network[:3],test_X)

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


# With log-likelihood the variance of the axes are taken into account
LL = LogLikelihood(training_errors)

# Score the training data. These should have a low mean score
training_ll_scores = LL.score(training_errors)
test_ll_scores     = LL.score(test_errors)

# The training scores should have a gamma distribution.
# Find the parameters describing the distribution
# Find a threshold that contains <threshold> of the training data
# I.E. 0.9 means 10% of the training data will be categorized as outliers

ll_thresh, ll_p = LL.gamma_threshold(threshold)

#ll_thresh, ll_p = gamma_threshold(training_ll_scores, threshold)

# Only the scores which surpasses the threshold will be considered an outlier
ll_predictions = [1 if x > ll_thresh else 0 for x in test_ll_scores]




# Some plotting
bins = np.linspace(1, 200, 150)
fig, ax = plt.subplots()
# Plot the training scores
ax.hist(training_ll_scores, bins, density=True)
# Plot the describing gamma distribution (not to scale)
min_ylim, max_ylim = plt.ylim()
ax.plot(bins, np.multiply(invgamma.pdf(bins, *ll_p), max_ylim*20))
plt.title("log likelihood training scores")
plt.show()

# Plot the ll scores
plot_scores(
        np.log(test_ll_scores), 
        test_Y, 
        'log(log likelihood scores)', 
        np.log(ll_thresh)
        )


# It seems r2 scoring performs better when we sample more data
print("r2 report:")
print(confusion_matrix(test_Y, r2_predictions))
print(classification_report(test_Y, r2_predictions))

# But LL has stable performance, even when undersampling
print("LL report:")
print(confusion_matrix(test_Y, ll_predictions))
print(classification_report(test_Y, ll_predictions))


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