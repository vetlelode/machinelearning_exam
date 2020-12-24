# Author: Ask H. B.

import pandas as pd
import numpy as np
from preprocessing import get_dataset, REAL_DATA_MAX_N
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.preprocessing import StandardScaler

from scipy.stats import invgamma


# Hyperparameters:
train_size    = 0.8 #0-1
pollution     = 0   #0-1
undersampling = 0.5 #0-1, though should not be below 1-threshold

hidden_layers = [10,10,2,10,10]
activation = "tanh" #or relu. Tanh works best (and gives the nicest graphs!)

# The percentile above which we can consider everything an outlier.
# Higher threshold means less false positives, but also less true negatives
threshold = 0.9


def relu(X):
    return np.vectorize(lambda x: x if x>0 else 0)(X)


def identity(x):
    return x


#Diagrams - Details unimportant
def plot_latent(network, Xs, cols=None, alphas=None, labels = None):
    latent = [encode(network[:3], X) for X in Xs]
    
    figure, ax = plt.subplots(figsize = (25,25))
    scatters = []
    for i in range(len(latent)):
        a = alphas[i] if alphas else 0.25
        c = cols[i] if cols else None
        s = ax.scatter(np.asarray(latent[i][:,0]),np.asarray(latent[i][:,1]), alpha=a, color=c)
        scatters.append(s)
    
    ax.legend(scatters, labels,scatterpoints=1, loc='lower left')
    plt.title('Latent Space', fontsize=15)
    plt.xlabel('Z1', fontsize=10)
    plt.ylabel('Z2', fontsize=10)
    plt.axis('equal')
    plt.show()


def encode(network, X):
    z = np.asmatrix(X)
    for weight, bias, activation in network:
        # sum the weights and then add the bias
        z = activation( np.add(np.matmul(z, weight), bias ) )
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
    
    
    def predict(self, xtest, threshold):

        # Score the training data. These should have a low mean score
        scores     = self.score(xtest)
        
        # The training scores should have a gamma distribution.
        # Find the parameters describing the distribution
        # Find a threshold that contains <threshold> of the training data
        # I.E. 0.9 means 10% of the training data will be categorized as outliers
        
        gamma_thresh, p = gamma_threshold(self.score(self.X), threshold)
        
        #aell_thresh, aell_p = gamma_threshold(training_ll_scores, threshold)
        
        # Only the scores which surpasses the threshold will be considered an outlier
        predictions = [1 if x > gamma_thresh else 0 for x in scores]
        
        return scores, gamma_thresh, predictions, p


def gamma_threshold(scores, threshold):
        # The density of values of the scores of the training data
        # follows the gamma distribution.
        # A sharp spike of likelihood followed by a sharp decline that 
        # smoothens out slowly.
        
        # In other words, it does not follow a normal distribution,
        # and as such, a more sohpisticated method to find the apropriate
        # threshold is warranted.
        
        # Calculate the describing parameters of the training data 
        # gamma distribution, and the threshold
        a, loc, scale = invgamma.fit(scores)
        return invgamma.isf(1-threshold, a, loc, scale), (a, loc, scale)

def plot_report(
        train_x, 
        inliers, 
        outliers, 
        p, 
        threshold, 
        title=None,
        xscale="linear",
        xaxis="score"):
    
    max_x = max(max(train_x), max(inliers), max(outliers))
    max_x = min(max_x, 8*np.std(outliers))
    min_x = min(min(train_x), min(inliers), min(outliers))
    
    # Plot the training scores
    bins = np.linspace(min_x, max_x, 150)
    if xscale == "log":
        bins = np.logspace(np.log10(min_x),np.log10(max_x),len(bins))
        
    
    fig, ax = plt.subplots(figsize=(15,15))
    ax.hist(
            x=(train_x, inliers), 
            bins=bins, 
            alpha=0.35,
            color=("blue","yellow"), 
            histtype="barstacked",
            density=True, 
            stacked=True, 
            label=("Training data","Inliers")
            )
    ax.hist(
            x=outliers, 
            bins=bins, 
            alpha=0.35, 
            color="black", 
            density=True, 
            stacked=True, 
            label="Outliers"
            )
    min_ylim, max_ylim = plt.ylim()
    ax.plot(bins, invgamma.pdf(bins, *p), label="Distribution curve of best fit")
    ax.vlines(threshold, min_ylim, max_ylim, color="black", label="Threshold")
    ax.legend(loc='upper left')
    ax.set_xscale(xscale)
    plt.xlabel(xaxis)
    plt.ylabel("Density")
    plt.title(title)
    plt.show()

def split_inliers_outliers(X,Y):
    inliers, outliers = [], []
    for i in range(len(X)):
        if Y[i]==0:
            inliers.append(X[i])
        else:
            outliers.append(X[i])
    return inliers, outliers

# MAIN PROGRAM


# Obtain the dataset
train_X, train_Y, test_X, test_Y = get_dataset(pollution=0, train_size=0.8)

undersample_idx = int(undersampling * len(train_X))
train_X = train_X[:undersample_idx]
train_Y = train_Y[:undersample_idx]

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

inliers, outliers = split_inliers_outliers(test_X,test_Y)

# Train a neural net into accurately recreate the input data 
# through a small latent space.
auto_encoder = MLPRegressor(
    solver="adam",
    activation=activation, 
    hidden_layer_sizes = hidden_layers,
    warm_start=False, #Used in debugging
    max_iter=50,
    verbose=True,
    tol=1e-7
    )


# The autoencoder is only being trained on inliers as to not learn to
# recreate outliers explicitly
auto_encoder.fit(train_X, train_X)


# Extract the network architecture from the autoencoder
n_layers = len(auto_encoder.coefs_)
activations = [np.tanh if activation =="tanh" else relu]*(n_layers-1)+[identity]
network = list(zip(auto_encoder.coefs_, auto_encoder.intercepts_, activations))

# Plot out the latent space (Pretty!)
plot_latent(network, Xs=(train_X,inliers,outliers), alphas=(0.2,0.3,0.5), cols=("blue","yellow","black"), labels=("training data","inliers","outliers"))


# Recreate the data from the auto encoder
train_recreations = auto_encoder.predict( train_X )
test_recreations  = auto_encoder.predict( test_X  )


# Score the samples with r2. This is the loss function used in the autoencoder.
# This is done because sklearn auto encoder does not support per sample scoring
#train_r2_scores = [r2_score( x_true, x_pred ) for x_true, x_pred in zip(train_X, train_recreations)]
train_r2_scores = [r2_score( train_X[i], train_recreations[i] ) for i in range(len(train_X))]
test_r2_scores = [r2_score( test_X[i], test_recreations[i] ) for i in range(len(test_X))]
#test_r2_scores  = [r2_score( x_true, x_pred ) for x_true, x_pred in zip(test_X, test_recreations)]


# Adjust the scores - Shift it so all the scores are positive
mm = max(train_r2_scores) + 0e-4  # add a small value to avoid dealing with zeroes
train_r2_scores = -(train_r2_scores - mm)
test_r2_scores = -(test_r2_scores - mm)

# Calculate the threshold, assuming the distribution of training scores
# follow the gamma distribution.
# We have seen that the gamma threshold gives a better scoring than
# that of a normal distribution threshold.

# We have observed that there is less overlap between the scores in r2
# and can therefore push the threshold further without much loss
r2_threshold, r2_p = gamma_threshold(train_r2_scores, 1-(1-threshold)/2)

# Predict the test data wrt the r2 threshold
r2_predictions = [1 if x > r2_threshold else 0 for x in test_r2_scores]

# Plot the gamma distribution of best fit
#plot_gamma(train_r2_scores, r2_p, r2_threshold, "training r2 scores")

#plot_scores(test_r2_scores, test_Y, 'R2 scores',r2_threshold)
plot_report(
        train_r2_scores, 
        *split_inliers_outliers(test_r2_scores,test_Y), 
        r2_p, 
        r2_threshold, 
        "R2"
        )

# R2 scoring does not take into account that the different axes
# might have different variances. Since the model is trained
# only on inliers, it might happen that the distinguishing
# axes of the outliers have low variance, and therefore R2 will
# not pick up on that.

# Take the error between the original data and its recreation
training_errors = train_X - train_recreations
test_errors     = test_X  - test_recreations


# With log-likelihood the variance of the axes are taken into account
# AELL - Auto-Encoder-Log-Likelihood
ae_LL = LogLikelihood(training_errors)

# Score the data.
training_aell_scores = ae_LL.score(training_errors)
test_aell_scores, aell_threshold, aell_predictions, aell_p = ae_LL.predict(test_errors,threshold)


# Plot the gamma distribution of best fit
plot_report(
        training_aell_scores, 
        *split_inliers_outliers(test_aell_scores,test_Y),
        aell_p, 
        aell_threshold, 
        title="Scoring by Log-Likelihood from AE-recreation-error",
        xaxis="score (log scale)",
        xscale="log"
        )


# Log-Likelihood is a statistical model we can use directly on the data as well,
# And we need to check if the Log-Likelihood performs as well, worse or better
# on the recreation error than on the raw data.
# If it performs as well or better, the AE is a redundant step.

direct_LL = LogLikelihood(train_X)

# Score the data. 
training_dll_scores = direct_LL.score(test_X)
test_dll_scores, dll_threshold, dll_predictions, dll_p = direct_LL.predict(test_X,threshold)

# Plot the gamma distribution of best fit
plot_report(
        training_dll_scores, 
        *split_inliers_outliers(test_dll_scores,test_Y),
        dll_p, 
        dll_threshold, 
        title="Log-Likelihood scores on the raw data",
        xaxis="score (log scale)",
        xscale="log"
        )


# It seems r2 scoring performs better when we sample more data
# The statistical model fitted from the training data, used to set a threshold,
# is not too good of a fit and produces more false negatives than we would
# like
print("r2 report:")
print(confusion_matrix(test_Y, r2_predictions))
print(classification_report(test_Y, r2_predictions))

# But AE-LL has stable performance, even when undersampling

# It seems the statistical model fitted on the training set
# is also a good fit for the testing data. There are about the
# same proportion of true false positives to true positives as
# false negatives to true negatives
print("AE-LL report:")
print(confusion_matrix(test_Y, aell_predictions))
print(classification_report(test_Y, aell_predictions))

# Direct LL gives comparably good results as AE-LL
# This means AE can be considered redundant wrt Log-Likelihood
print("direct-LL report:")
print(confusion_matrix(test_Y, dll_predictions))
print(classification_report(test_Y, dll_predictions))


# Covariance analysis

inlier_errors, outlier_errors = split_inliers_outliers(test_errors,test_Y)

# The recreation error of inliers are for the mostly uncorrelated
plt.matshow(pd.DataFrame(inlier_errors).corr())
plt.title("Correlation matrix: inlier errors")
plt.show()

plt.matshow(pd.DataFrame(outlier_errors).corr())
plt.title("Correlation matrix: outlier errors")
plt.show()

plt.matshow(pd.DataFrame(training_errors).corr())
plt.title("Correlation matrix: training errors")
plt.show()



"""    
reduced_test_X = test_X[:,most_significant_axes]
reduced_test_recreations = test_recreations[:,most_significant_axes]
adjtest_r2_scores  = [r2_score( x_true, x_pred ) for x_true, x_pred in zip(reduced_test_X, reduced_test_recreations)]
"""