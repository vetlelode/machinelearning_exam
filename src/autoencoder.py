import pandas as pd
import numpy as np
from preprocessing import get_dataset, REAL_DATA_MAX_N
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.preprocessing import StandardScaler

from scipy.stats import invgamma


#Hyperparameters:
training_size = REAL_DATA_MAX_N//10 #284315
hidden_layers = [20,10,2,10,20]
activation = "tanh" #or relu. Tanh works best (and giges the nicest graphs!)
# Higher threshold means less false positives, but also less true negatives
threshold = 0.9

# max 30
axes_of_importance = 15

# sort axes by variance    : variance
# sort axes by correlation : correlation
axes_relation = "correlation"

# use lowest var/corr axes : low
# use highest var/corr axes: high
axis_sort = "low"

# low correlation axes work well

def relu(X):
    return np.vectorize(lambda x: x if x>0 else 0)(X)


def identity(x):
    return x


#Diagrams - Details unimportant
def plot_histogram(X, Y, title, threshold : float = None) -> None:
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
    max_x = min(max(scores)*1.5, np.mean(scores)+3*np.std(scores))
    bins = np.linspace(0, max_x, 250)
    fig, ax = plt.subplots(figsize=(15,15))
    # Plot the training scores
    ax.hist(scores, bins, density=True, stacked=True)
    # Plot the describing gamma distribution (not to scale)
    min_ylim, max_ylim = plt.ylim()
    ax.plot(bins, invgamma.pdf(bins, *p))
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
    plot_scatter  ( np.log(scores), Y, f"log({title})", np.log(threshold))


def sort_axes_by_correlation(corr_basis):
    corr = pd.DataFrame(corr_basis).corr()
    axis_importance = [(sum(c**2),i) for c,i in zip(np.asarray(corr),range(corr.shape[0]))]
    most_significant_axes = np.asarray(sorted(axis_importance)[-1:0:-1])[:,1]
    # for some reason, the indices are casted to float, which cannot be used as
    # indices
    return np.vectorize(int)(most_significant_axes)

def sort_axes_by_variance(var_basis):
    var_basis = np.asmatrix(var_basis)
    axis_importance = [(np.var(var_basis[:,i]),i) for i in range(var_basis.shape[1])]
    most_significant_axes = np.asarray(sorted(axis_importance)[-1:0:-1])[:,1]
    # for some reason, the indices are casted to float, which cannot be used as
    # indices
    return np.vectorize(int)(most_significant_axes)

# MAIN PROGRAM


# Obtain the dataset
train_X, train_Y, test_X, test_Y = get_dataset(k1=training_size,f=0)

most_correlated_axes = sort_axes_by_variance(train_X) if axes_relation == "variance" else sort_axes_by_correlation(train_X)

if axis_sort == "high":
    train_X = np.asmatrix(train_X)[:,most_correlated_axes[-axes_of_importance:]]
    test_X = np.asmatrix(test_X)[:,most_correlated_axes[-axes_of_importance:]]
else:
    train_X = np.asmatrix(train_X)[:,most_correlated_axes[:axes_of_importance]]
    test_X = np.asmatrix(test_X)[:,most_correlated_axes[:axes_of_importance]]

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

inliers  = [x for x,y in zip(test_X,test_Y) if y==0]
outliers = [x for x,y in zip(test_X,test_Y) if y==1]

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
plot_latent(network, Xs=(train_X,inliers,outliers), alphas=(0.2,0.5,0.5), cols=("blue","green","black"), labels=("training data","inliers","outliers"))


# Recreate the data from the auto encoder
train_recreations = auto_encoder.predict( train_X )
test_recreations  = auto_encoder.predict( test_X  )


# Score the samples with r2. This is the loss function used in the autoencoder.
# This is done because sklearn auto encoder does not support per sample scoring
train_r2_scores = [r2_score( x_true, x_pred ) for x_true, x_pred in zip(train_X, train_recreations)]
test_r2_scores  = [r2_score( x_true, x_pred ) for x_true, x_pred in zip(test_X, test_recreations)]


# Adjust the scores - Shift it so all the scores are positive
mm = max(train_r2_scores)
train_r2_scores = -(train_r2_scores - mm)
test_r2_scores = -(test_r2_scores - mm)

# Calculate the threshold, assuming the distribution of training scores
# follow the gamma distribution.
# We have seen that the gamma threshold gives a better scoring than
# that of a normal distribution threshold.
r2_threshold, r2_p = gamma_threshold(train_r2_scores,threshold)

# Predict the test data wrt the r2 threshold
r2_predictions = [1 if x > r2_threshold else 0 for x in test_r2_scores]

# Plot the gamma distribution of best fit
plot_gamma(train_r2_scores, r2_p, r2_threshold, "training r2 scores")
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
# AELL - Auto-Encoder-Log-Likelihood
ae_LL = LogLikelihood(training_errors)

# Score the data.
training_aell_scores = ae_LL.score(training_errors)
test_aell_scores, aell_threshold, aell_predictions, aell_p = ae_LL.predict(test_errors,threshold)


# Plot the gamma distribution of best fit
plot_gamma( training_aell_scores, aell_p, aell_threshold, "training AE-LL scores" )
plot_scores( test_aell_scores, test_Y, 'AE log-likelihood scores', aell_threshold )


# Log-Likelihood is a statistical model we can use directly on the data as well,
# And we need to check if the Log-Likelihood performs as well, worse or better
# on the recreation error than on the raw data.
# If it performs as well or better, the AE is a redundant step.

direct_LL = LogLikelihood(train_X)

# Score the data. 
training_dll_scores = ae_LL.score(training_errors)
test_dll_scores, dll_threshold, dll_predictions, dll_p = ae_LL.predict(test_X,threshold)

# Plot the gamma distribution of best fit
plot_gamma(training_dll_scores, dll_p, dll_threshold, "Direct Log Likelihood scores", 20)
plot_scores( test_dll_scores, test_Y, 'direct log-likelihood scores)', dll_threshold )

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

# Direct LL gives usable results, but are consistently worse than when applied
# to the recreation error
print("direct-LL report:")
print(confusion_matrix(test_Y, dll_predictions))
print(classification_report(test_Y, dll_predictions))


# Covariance analysis

inlier_errors = [x for x,y in zip(test_errors,test_Y) if y==0]
outlier_errors = [x for x,y in zip(test_errors,test_Y) if y==1]

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