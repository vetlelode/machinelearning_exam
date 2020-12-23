import pandas as pd
import numpy as np
from preprocessing import get_dataset
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.preprocessing import StandardScaler

from scipy.stats import invgamma


#Hyperparameters:
hidden_layers = [20,20,10,2,10,20,20]
components = 28
activation = "tanh" #or relu
# Higher threshold means less false positives, but also less true negatives
threshold = 0.9


def relu(X):
    return np.vectorize(lambda x: x if x>0 else 0)(X)


def identity(x):
    return x


#Diagrams - Details unimportant
def plot_histogram(X, Y, title, threshold=None) -> None:
    #Omit extreme outliers
    std = np.std(X)
    mean = np.mean(X)
    lower_bound, upper_bound = max(min(X), mean-2*std), min(max(X),mean+2*std)
    
    X_inliers = [x for x,y in zip(X,Y) if y==0 ]
    X_outliers = [x for x,y in zip(X,Y) if y==1 ]
    
    bins = np.linspace(lower_bound, upper_bound, 250)
    
    fig, ax = plt.subplots(figsize=(15,15))
    ax.hist(x=(X_inliers, X_outliers), bins=bins, alpha=0.5, label=('inliers','outliers'), stacked=False, histtype="stepfilled", density=True)
    ax.legend(loc='upper left')
    if threshold != None:
        min_ylim, max_ylim = plt.ylim()
        ax.vlines(threshold,min_ylim,max_ylim)
    plt.title(title)
    plt.show()


#Diagrams - Details unimportant
def plot_scatter(X,Y,title,threshold=None) -> None:
    fig, ax = plt.subplots(figsize = (15,15))
    
    scatter = ax.scatter(range(len(X)),X,c=Y, alpha=0.3)
    handles, labels = scatter.legend_elements()
    ax.legend(handles, ["inliers", "outliers"], loc='lower left')
    if threshold != None:
        min_xlim, max_xlim = plt.xlim()
        ax.hlines(threshold,min_xlim,max_xlim)
    plt.title(title)
    plt.show()

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

def plot_gamma(scores, p, threshold, title, scale=0.3):
    #
    max_x = min(max(scores), np.mean(scores)+3*np.std(scores))
    bins = np.linspace(0, max_x, 150)
    fig, ax = plt.subplots(figsize=(15,15))
    # Plot the training scores
    ax.hist(scores, bins, density=True)
    # Plot the describing gamma distribution (not to scale)
    min_ylim, max_ylim = plt.ylim()
    ax.plot(bins, np.multiply(invgamma.pdf(bins, *p), max_ylim*scale))
    ax.vlines(threshold, min_ylim, max_ylim, color="black")
    plt.title(title)
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

def plot_scores(scores, Y, title, threshold=None):
    plot_histogram( scores, Y, title, threshold)
    plot_scatter  ( scores, Y, title, threshold)


# Obtain the dataset
train_X, train_Y, test_X, test_Y = get_dataset(k1=8000,f=0)

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

activations = [np.tanh if activation =="tanh" else relu]*(n_layers-1)+[identity]
network = list(zip(auto_encoder.coefs_, auto_encoder.intercepts_, activations))

inliers  = [x for x,y in zip(test_X,test_Y) if y==0]
outliers = [x for x,y in zip(test_X,test_Y) if y==1]

plot_latent(network, Xs=(train_X,inliers,outliers), alphas=(0.2,0.5,0.5), cols=("blue","green","black"), labels=("training data","inliers","outliers"))


# Recreate the data from the auto encoder
train_recreations = auto_encoder.predict( train_X )
test_recreations  = auto_encoder.predict( test_X  )

# Score the samples with r2. This is the loss function used in the autoencoder.
# This is done because sklearn auto encoder does not support per sample scoring
train_r2_scores = [r2_score( x_true, x_pred ) for x_true, x_pred in zip(train_X, train_recreations)]
test_r2_scores  = [r2_score( x_true, x_pred ) for x_true, x_pred in zip(test_X, test_recreations)]


mm = max(train_r2_scores)
train_r2_scores = -(train_r2_scores - mm)
test_r2_scores = -(test_r2_scores - mm)

r2_threshold, r2_p = gamma_threshold(train_r2_scores,threshold)

plot_gamma(train_r2_scores, r2_p, r2_threshold, "training r2 scores", 0.3)

# The scores from the training follows the gamma distribution, but the ones from



# Find a decent threshold to split inliers and outliers
# Use the 68% rule (1 standard deviation)
#r2_threshold = np.mean(train_r2_scores) + np.std(train_r2_scores)

# Predict the test data wrt the r2 threshold
r2_predictions = [1 if x > r2_threshold else 0 for x in test_r2_scores]


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
ae_LL = LogLikelihood(training_errors)

# Score the training data. These should have a low mean score
training_aell_scores = ae_LL.score(training_errors)
test_aell_scores, aell_threshold, aell_predictions, aell_p = ae_LL.predict(test_errors,threshold)


# Some plotting
plot_gamma(training_aell_scores, aell_p, aell_threshold, "training AE-LL scores", 20)

# Plot the ll scores
plot_scores(
        np.log(test_aell_scores), 
        test_Y, 
        'log(log AE likelihood scores)', 
        np.log(aell_threshold)
        )


direct_LL = LogLikelihood(train_X)

# Score the training data. These should have a low mean score
training_dll_scores = ae_LL.score(training_errors)
test_dll_scores, dll_threshold, dll_predictions, dll_p = ae_LL.predict(test_X,threshold)

plot_gamma(training_dll_scores, dll_p, dll_threshold, 20)
plot_scores(
        np.log(test_dll_scores), 
        test_Y, 
        'log(log direct likelihood scores)', 
        np.log(dll_threshold)
        )

# It seems r2 scoring performs better when we sample more data

print("r2 report:")
print(confusion_matrix(test_Y, r2_predictions))
print(classification_report(test_Y, r2_predictions))


# But AE-LL has stable performance, even when undersampling
print("AE-LL report:")
print(confusion_matrix(test_Y, aell_predictions))
print(classification_report(test_Y, aell_predictions))

# Direct LL gives usable results, but perform poorer with more data
print("direct-LL report:")
print(confusion_matrix(test_Y, dll_predictions))
print(classification_report(test_Y, dll_predictions))

inlier_errors = [x for x,y in zip(test_errors,test_Y) if y==0]
outlier_errors = [x for x,y in zip(test_errors,test_Y) if y==1]

# The recreation error of inliers are for the mostly uncorrelated
plt.matshow(pd.DataFrame(inlier_errors).corr())
plt
plt.title("Correlation matrix: inlier errors")
plt.show()

plt.matshow(pd.DataFrame(outlier_errors).corr())
plt.title("Correlation matrix: outlier errors")
plt.show()

plt.matshow(pd.DataFrame(training_errors).corr())
plt.title("Correlation matrix: training errors")
plt.show()