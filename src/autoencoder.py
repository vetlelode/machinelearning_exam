# Author: Ask H. B.

import pandas as pd
import numpy as np
from preprocessing import get_dataset
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.preprocessing import StandardScaler

from scipy.stats import invgamma

# Hyperparameters:
train_size    = 0.8 #0-1
pollution     = 0   #0-1

undersampling = 100_000

hidden_layers = [10,10,2,10,10]
activation = "tanh" #or relu. Tanh works best (and gives the nicest graphs!)

# The percentile above which we can consider everything an outlier.
# Higher threshold means less false positives, but also less true negatives

# R2 works best with a threshold of 0.9
# LL works better with a threshold of 0.99
threshold = 0.99


def relu(X):
    return np.vectorize(lambda x: max(0,x))(X)


def identity(x):
    return x


def encode(network, X):
    z = np.asmatrix(X)
    for weight, bias, activation in network:
        # sum the weights and then add the bias
        z = activation( np.add(np.matmul(z, weight), bias ) )
    return np.asarray(z)


class LogLikelihood:
    
    def __init__(self, X, threshold=None):
        X = np.asmatrix(X)
        self.n_axes = X.shape[1]
        # Create a n-dimensional normal distribution:
        
        # log likelihood parameters
        # the mean of each axis
        self.μs = [np.mean(X[:,i]) for i in range(self.n_axes)]
        # the standard deviation of each axis
        self.σ2s = [np.var(X[:,i]) for i in range(self.n_axes)]
        self.train_scores = self.score(X)
        
        # Since the log-likelihood is the sum of squares,
        # the probability distribution follows a chi2 distribution,
        # which is a special case of the gamma distribution.
        
        # Calculate the describing parameters of the distribution
        self.p = invgamma.fit(self.train_scores)
        
        # Calculate the threshold if one isn't provided
        if threshold is not None:
            self.set_threshold(threshold)
    
    
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
    
    
    def predict(self, xtest, threshold=None):
        scores = self.score(xtest)
        return self.predict_from_scores(scores, threshold)
    
    def predict_from_scores(self, scores, threshold=None):
        if threshold is not None:
            self.set_threshold(threshold)
        # If score exceeds the threshold, it is identified as an outlier
        return [1 if x > self.threshold else 0 for x in scores]
    
    def set_threshold(self, threshold):
        self.threshold, _ = gamma_threshold(self.train_scores, threshold, p=self.p)


class AutoEncoderOutlierPredictor:
    def __init__(
            self,
            hidden_layers: list = [20,10,2,10,10,20],
            activation : str ="tanh",
            threshold = 0.9,
            verbose=True,
            max_iter=50
            ):
        self.auto_encoder = MLPRegressor(
            solver="adam",
            activation=activation, 
            hidden_layer_sizes = hidden_layers,
            warm_start=False, #Used in debugging
            max_iter=max_iter,
            verbose=verbose,
            tol=1e-7
            )
        self.verbose = verbose
        self.layers = len(hidden_layers)
        self.activation = activation
        self.threshold = threshold
    
    
    def fit(self, train_X):
        print("fitting data")
        # Scale data to make training easier
        # -
        # We tried to normalize and scale, but that made it *too*
        # easy to train, and we got extreme overfitting and the model
        # would fit new inliers and outliers equaly poorly.
        # A way to avoid overfitting would be to introduce noise to the training data
        # per epoch, but sklearn does not support this.
        # Standardscaler works fairly well on its own.
        self.scaler = StandardScaler().fit(train_X)
        self.train_X = self.scaler.transform(train_X)
        
        # The autoencoder is only being trained on inliers as to not learn to
        # recreate outliers explicitly
        self.auto_encoder.fit(self.train_X, self.train_X)
        self.train_recreation = self.auto_encoder.predict(self.train_X)
        
        
        # R2
        if self.verbose:
            print("Scoring r2 data")
        
        # Score the training data with r2, which is the loss function used in
        # the auto encoder. Outliers should present worse scoring in the loss
        # function, as the auto encoder has never been trained on them
        train_r2_scores = self._score_r2(self.train_X, self.train_recreation)
        
        # Adjust the scoring to be positive
        mm = max(train_r2_scores) + 1e-7  # add a small value to avoid dealing with zeroes
        self.r2_transform = lambda scores: -(scores-mm)
        self.train_r2_scores = self.r2_transform(train_r2_scores)
        # The r2 scores may not follow a gamma distribution entirely,
        # but in our experience, it follows it well enough to give us
        # decent results, and we will continue to assume it is
        # following a gamma distribution
        
        # No significant portion of outliers present themselves on
        # the lower end of the distribution, so it is reasonable
        # to set the threshold on only one side of the curve
        self.r2_threshold, self.r2_p = gamma_threshold(self.train_r2_scores, self.threshold)
        # p is the describing parameters of the gamma curve
        
        # Log-Likelihood
        if self.verbose:
            print("scoring log-likelihood data")
        
        # R2 performs reasonably, but the difference between inliers
        # and outliers is small. Some axes have higher variance than others
        # in the training data, and those axes will have a bigger effect
        # on the score than the axes with lower variance. Since the model
        # has not been trained on any outliers, it is reasonable to assume
        # the variances on their axes will be different.
        # In order to detect outliers, the variances of each axis should be
        # taken into account. Log likelihood is a method to give a statistical
        # score of how likely a n-dimensional data point is to be found
        # at that specific point. It is assuming each axes has a normal 
        # distribution, and that the likelihood of that specific point
        # is the product of the likelihoods of each axis. The log likelihood
        # is just the logarithm of that likelihood.
        # This is useful as instead of taking the product - which may lead
        # to rounding errors - we take the sum. The constant factors can also
        # be dropped for a decently fast scoring.
        
        # We are assuming that each axis follows more or less a normal
        # distribution, and therefore the sum of its squares will be
        # chi2 distributed (with ~30 degrees of freedom).
        # Since not all axes are normally distributed, we instead
        # assume the log-likelihood follows the more generalized gamma 
        # distribution
        train_recreation_errors = self.train_X-self.train_recreation
        self.LL = LogLikelihood(train_recreation_errors, self.threshold)
        
        print("Fit complete")
        
    
    def forward_propogate(self, *datas, n_layers : int):
        # Scale data to the same space as the training data
        datas = [self.scaler.transform(data) for data in datas]
        
        # populate the network with activation functions
        activations = [np.tanh if self.activation =="tanh" else relu]*(self.layers)+[identity]
        network = list(zip(self.auto_encoder.coefs_, self.auto_encoder.intercepts_, activations))
        
        # Encode n layers of the network
        return [encode(network[:n_layers], data) for data in datas]
    
    
    def _score_r2(self, data, recreated_data):
        # Score the samples with r2. This is the loss function used in the 
        # autoencoder. This is done because sklearn auto encoder does not 
        # support per sample scoring
        return [r2_score( data[i], recreated_data[i] ) for i in range(len(data))]
    
    
    def score_r2(self, data):
        # Scale data to the same space as the training data
        data = self.scaler.transform(data)
        # Score the samples with r2. This is the loss function used in the autoencoder.
        scores = self._score_r2(data, self.auto_encoder.predict(data))
        # Transform the scores such that most of it is positive
        return self.r2_transform(scores)
    
    def predict_r2(self, data):
        # Scale data to the same space as the training data
        data = self.scaler.transform(data)
        # Score the samples with r2. This is the loss function used in the autoencoder.
        scores = self.score_r2(data)
        return self.predict_r2_from_scores(scores)
    
    
    def predict_r2_from_scores(self, scores):
        # If scores exceed the threshold, it is identified as an outlier
        return [1 if x > self.r2_threshold else 0 for x in scores]
    
    
    def score_ll(self, data):
        data = self.scaler.transform(data)
        return self.LL.score(data)
    
    
    def predict_ll(self, data):
        data = self.scaler.transform(data)
        return self.LL.predict(data)
    
    
    def predict_ll_from_scores(self, scores):
        return self.LL.predict_from_scores(scores)


def gamma_threshold(scores, threshold, p=None):
        # The density of values of the scores of the training data
        # follows the gamma distribution.
        # This is what we would expect from the sum of stochastich
        # variables squared, which is what the scoring functions
        # are.
        
        # Calculate the describing parameters of the training data 
        # gamma distribution, and the threshold
        if p is None:
            p = invgamma.fit(scores)
        a, loc, scale = p
        return invgamma.isf(1-threshold, a, loc, scale), (a, loc, scale)


#Diagrams - Details unimportant
def plot_latent(latents, cols=None, alphas=None, labels = None):
    
    figure, ax = plt.subplots(figsize = (25,25))
    scatters = []
    for i in range(len(latents)):
        a = alphas[i] if alphas else 0.25
        c = cols[i] if cols else None
        s = ax.scatter(np.asarray(latents[i][:,0]),np.asarray(latents[i][:,1]), alpha=a, color=c)
        scatters.append(s)
    
    ax.legend(scatters, labels,scatterpoints=1, loc='lower left')
    plt.title('Latent Space', fontsize=15)
    plt.xlabel('Z1', fontsize=10)
    plt.ylabel('Z2', fontsize=10)
    plt.axis('equal')
    plt.show()


# Plotting. Details unimportant
def plot_report(
        train_x, 
        inliers, 
        outliers, 
        p, 
        threshold, 
        title=None,
        xscale="linear",
        xaxis="score"):
    
    X = (train_x, inliers, outliers)
    max_x = max(np.vectorize(max)(X))
    min_x = min(np.vectorize(min)(X))
    
    
    # Plot the training scores
    n_bins = 150
    if xscale == "log":
        bins = np.logspace(np.log10(min_x),np.log10(max_x),n_bins)
    else:
        bins = np.linspace(min_x, max_x, n_bins)
    
    
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_axes([0.0, 0.5, 0.8, 0.4])
    ax2 = fig.add_axes([0.0, 0.1, 0.8, 0.4])
    ax3 = ax2.twinx()
    ax4 = ax1.twinx()
    
    histogram = lambda ax, X, color, density, label: ax.hist(
        x=X, 
        bins=bins, 
        alpha=0.35,
        color=color, 
        histtype="barstacked",
        density=density,
        stacked=True,
        label=label
        )
    histogram( ax1, train_x,  "blue",   True , "training data")
    histogram( ax2, inliers,  "yellow", True , "inliers" )
    histogram( ax3, outliers, "black",  False, "outliers" )
    
    gamma = invgamma.pdf(bins, *p)
    
    # A lot of hacky stuff here
    
    # this curve is a bit bugged, adjusting the height fixes it
    ax4.set_ylim(0,max(gamma))
    ax4.plot(bins, gamma)
    
    [ax.set_xscale(xscale) for ax in (ax1,ax2,ax3,ax4)]
    
    # Ignore this, the library is being obstinate
    miny, maxy = plt.ylim()
    ax1.vlines(threshold, miny, maxy, color="black")
    ax2.vlines(threshold, miny, maxy, color="black")
    ax1.legend( ("training data","threshold"), loc="upper right")
    ax2.legend( ("inliers", "threshold"), loc="upper right")
    ax4.legend( ["Gamma curve of best fit"], loc="center right")
    ax3.legend( ["outliers"], loc="center right")
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

train_X, train_Y, test_X, test_Y = get_dataset(
        sample     = undersampling,
        # Not training the encoder on any outliers gives the best results
        pollution  = 0, # How much of the outliers to put in the training set
        train_size = 0.8 # How much of the inliers to put in the training set
        )


# Isolate inliers and outliers for graphing
inliers, outliers = split_inliers_outliers(test_X,test_Y)

# Set up the autoencoder
AE = AutoEncoderOutlierPredictor(
        hidden_layers = hidden_layers,
        activation = activation,
        threshold = threshold
        )

# Fit to training data, this will take a while
AE.fit(train_X)

# Forward propogate to the latent space
train_latent, inlier_latent, outlier_latent = AE.forward_propogate(train_X, inliers, outliers, n_layers=3)

# Plot out the latent space (Pretty!)
plot_latent(
        (train_latent, inlier_latent, outlier_latent), 
        alphas=(0.2,0.3,0.5), 
        cols=("blue","yellow","black"), 
        labels=("training data","inliers","outliers")
        )

# Score with R2, the loss function of the regressor
r2_scores = AE.score_r2(test_X)
# Predict its class from the score
r2_pred = AE.predict_r2_from_scores(r2_scores)

# Plotting
plot_report(
        AE.train_r2_scores, 
        *split_inliers_outliers(r2_scores,test_Y), 
        AE.r2_p, 
        AE.r2_threshold, 
        "R2"
        )

# R2 scoring does not take into account that the different axes
# might have different variances. Since the model is trained
# only on inliers, it might happen that the distinguishing
# axes of the outliers have low variance, and therefore R2 will
# not pick up on that.

# Score based on the log-likelihood of the recreation errors
aell_scores = AE.score_ll(test_X)
# Predict its class from the score
aell_pred = AE.predict_ll_from_scores(aell_scores)

# Plotting
plot_report(
        AE.LL.train_scores, 
        *split_inliers_outliers(aell_scores,test_Y),
        AE.LL.p, 
        AE.LL.threshold, 
        title="Scoring by Log-Likelihood from AE-recreation-error",
        xaxis="score (log scale)",
        xscale="log"
        )


# Log-Likelihood is a statistical model we can use directly on the data as well,
# And we need to check if the Log-Likelihood performs as well, worse or better
# on the recreation error than on the raw data.
# If it performs as well or better, the AE is a redundant step.
direct_LL = LogLikelihood(train_X, threshold)

# Score the data. 
dll_scores = direct_LL.score(test_X)
# Predict its class from the score
dll_pred = direct_LL.predict_from_scores(dll_scores)

# Plotting
plot_report(
        direct_LL.train_scores, 
        *split_inliers_outliers(dll_scores,test_Y),
        direct_LL.p, 
        direct_LL.threshold, 
        title="Log-Likelihood scores on the raw data",
        xaxis="score (log scale)",
        xscale="log"
        )


# It seems r2 scoring performs better when we sample more data
# The statistical model fitted from the training data, used to set a threshold,
# is not too good of a fit and produces more false negatives than we would
# like.

# R2 has a narrower gap between outliers and inliers, and a lot more overlap.
# This leads to more uncertain detection, even for more extreme outliers.
# Considering that different scores could be used differently in an operational
# environment, such as blocking all transactions above a certain threshold,
# while scores under that one but above another be put to manual
# inspection, we would like to minimize this gap where manual
# inspection is needed.
print("r2 report:")
print(confusion_matrix(test_Y, r2_pred))
print(classification_report(test_Y, r2_pred))

# AE-LL has stable performance, even when undersampling
# Log-Likelihood of the reconstruction errors provide better and more
# certain classifications. The overlapping section between outliers and inliers
# is noticeably narrowed, as outliers have more extreme scores.
# This is the better option with respect to the operational.
print("AE-LL report:")
print(confusion_matrix(test_Y, aell_pred))
print(classification_report(test_Y, aell_pred))

# Log-likelihood of the raw data gives comparable results
# to the auto-encoder recreation error log likelihood.
# This is probably because the inliers of the raw data
# are pretty evenly distributed, having low correlation,
# while the outliers have a significant correlation.
# This shows that a standard statistical model could be used
# for satisfactory results, as no advanced non-linear model
# is needed.
print("direct-LL report:")
print(confusion_matrix(test_Y, dll_pred))
print(classification_report(test_Y, dll_pred))


# Correlation analysis

# The recreation error of inliers are for the mostly uncorrelated
plt.matshow(pd.DataFrame(inliers).corr())
plt.title("Correlation matrix: inliers")
plt.show()

plt.matshow(pd.DataFrame(outliers).corr())
plt.title("Correlation matrix: outliers")
plt.show()

plt.matshow(pd.DataFrame(train_X).corr())
plt.title("Correlation matrix: training data")
plt.show()
