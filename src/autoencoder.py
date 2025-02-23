import pandas as pd # Only used in multicollinearity analysis
import numpy as np # Used for all sorts of matrix operations

import matplotlib.pyplot as plt # plotting...

# Neural net for the autoencoder
from sklearn.neural_network import MLPRegressor
# Admittedly a deceptive metric for scoring highly unbalanced datasets
from sklearn.metrics import confusion_matrix
# Some important metrics to look at when grading our models
from sklearn.metrics import classification_report
# Scales the data to minimize the impact of high variance features
from sklearn.preprocessing import StandardScaler
# Visualizes the data in two dimensions while preserving a sense of nearnes
from sklearn.manifold import TSNE


# Custom libraries

from stat_tools import gamma_threshold
from stat_tools import LogLikelihood
from stat_tools import split_inliers_outliers
from stat_tools import OutlierDetectorScorer

from preprocessing import get_dataset
from plotting import plot_report
from plotting import scatterplot
from plotting import prc_plot

# Hyperparameters:
train_size    = 0.5 # 0-1
pollution     = 0    # 0-1

undersampling = 240_000
hidden_layers = [20, 10, 2, 10, 20] # Latent space works poorly at size=1
activation = "tanh"  # or relu. Tanh works best (and gives the nicest graphs!)

# The percentile above which we can consider everything an outlier.
# Higher threshold means less inliers detected as outliers,
# but more outliers detected as inliers.
threshold = 0.99
l2_threshold = 0.996
ll_threshold = 0.99


def relu(X):
    return np.vectorize(lambda x: max(0, x))(X)


def identity(x):
    return x

# Forward propogation through a custom number of layers
# Used in order to visualize the latent space.
def encode(network, X):
    z = np.asmatrix(X)
    for weight, bias, activation in network:
        # sum the weights and then add the bias
        z = activation( np.add(np.matmul(z, weight), bias ) )
    return np.asarray(z)

# Our own autoencoder based off of sklearns MLPRegressor
class AutoEncoderOutlierPredictor:
    def __init__(
            self,
            hidden_layers: list = [20, 10, 2, 10, 10, 20],
            activation : str = "tanh",
            threshold = 0.9,
            l2_threshold = None,
            ll_threshold = None,
            verbose=True,
            max_iter=50
            ):
        self.auto_encoder = MLPRegressor(
            solver="adam",
            activation=activation, 
            hidden_layer_sizes=hidden_layers,
            warm_start=False,  # Used in debugging
            max_iter=max_iter,
            verbose=verbose,
            tol=1e-7
            )
        self.verbose = verbose
        self.layers = len(hidden_layers)
        self.activation = activation
        self.threshold = threshold
        self.l2_threshold = l2_threshold if l2_threshold else threshold
        self.ll_threshold = ll_threshold if ll_threshold else threshold

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
        
        
        # l2
        if self.verbose:
            print("Scoring l2 data")
        
        # Score the training data with l2, which is the loss function used in
        # the auto encoder. Outliers should present worse scoring in the loss
        # function, as the auto encoder has never been trained on them
        self.train_l2_scores = l2_score(self.train_X, self.train_recreation)
        
        # No significant portion of outliers present themselves on
        # the lower end of the distribution, so it is reasonable
        # to set the threshold on only one side of the curve
        self.l2_threshold, self.l2_p = gamma_threshold(self.train_l2_scores, self.l2_threshold)
        # p is the describing parameters of the gamma curve
        
        # Log-Likelihood
        if self.verbose:
            print("scoring log-likelihood data")
        
        # Some axes have higher variance than others
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
        self.LL = LogLikelihood(train_recreation_errors, self.ll_threshold)
        
        print("Fit complete")

    def forward_propogate(self, *datas, n_layers : int):
        # Scale data to the same space as the training data
        datas = [self.scaler.transform(data) for data in datas]
        
        # populate the network with activation functions
        activations = [np.tanh if self.activation =="tanh" else relu]*(self.layers)+[identity]
        network = list(zip(self.auto_encoder.coefs_, self.auto_encoder.intercepts_, activations))
        
        # Encode n layers of the network
        return [encode(network[:n_layers], data) for data in datas]

    def score_l2(self, data):
        # Scale data to the same space as the training data
        data = self.scaler.transform(data)
        # Score the samples with l2. This is the loss function used in the autoencoder.
        scores = l2_score(data, self.auto_encoder.predict(data))
        # Transform the scores such that most of it is positive
        return scores
    
    def predict_l2(self, data):
        # Scale data to the same space as the training data
        data = self.scaler.transform(data)
        # Score the samples with l2. This is the loss function used in the autoencoder.
        scores = self.score_l2(data)
        return self.predict_l2_from_scores(scores)

    def predict_l2_from_scores(self, scores):
        # If scores exceed the threshold, it is identified as an outlier
        return [1 if x > self.l2_threshold else 0 for x in scores]

    def score_ll(self, data):
        data = self.scaler.transform(data)
        return self.LL.score(data)
    
    def predict_ll(self, data):
        data = self.scaler.transform(data)
        return self.LL.predict(data)

    def predict_ll_from_scores(self, scores):
        return self.LL.predict_from_scores(scores)

def l2_score(y_true, y_pred):
    # Loss squared. The loss function of the autoencoder.
    return np.square(np.subtract(y_true, y_pred)).sum(axis=1) #sum the row.


# MAIN PROGRAM


# Obtain the dataset

train_X, train_Y, test_X, test_Y = get_dataset(
        sample=undersampling,
        # Not training the encoder on any outliers gives the best results
        pollution=pollution,  # How much of the outliers to put in the training set
        train_size=train_size  # How much of the inliers to put in the training set
        )

# Isolate inliers and outliers for graphing
inliers, outliers = split_inliers_outliers(test_X, test_Y)


# Set up the autoencoder
AE = AutoEncoderOutlierPredictor(
        hidden_layers=hidden_layers,
        activation=activation,
        threshold=threshold,
        l2_threshold=l2_threshold,
        ll_threshold=ll_threshold
        )

# Fit to training data, this will take a while
AE.fit(train_X)

# Forward propogate to the latent space
train_latent, inlier_latent, outlier_latent = AE.forward_propogate(train_X, inliers, outliers, n_layers=3)

# Plot out the latent space (Pretty!)
scatterplot(
        (train_latent, inlier_latent, outlier_latent), 
        alphas=(0.2, 0.3, 0.5),
        cols=("blue", "yellow", "black"),
        labels=("training data", "inliers", "outliers"),
        title="Latent space"
        )

# Score with l2, the loss function of the regressor
l2_scores = AE.score_l2(test_X)
# Predict its class from the score
l2_pred = AE.predict_l2_from_scores(l2_scores)

# Plotting

plot_report(
        AE.train_l2_scores, 
        *split_inliers_outliers(l2_scores, test_Y),
        AE.l2_p, 
        AE.l2_threshold, 
        "l2",
        xaxis="score (log scale)",
        xscale="log"
        )

# l2 scoring does not take into account that the different axes
# might have different variances. Since the model is trained
# only on inliers, it might happen that the distinguishing
# axes of the outliers have low variance, and therefore l2 will
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

# Scaling the data leads to the same results as not scaling it.
direct_LL = LogLikelihood(train_X, ll_threshold)

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

# The baseline to compare the AUPRC scores to
baseline = sum(test_Y)/len(test_Y)

# l2 has a narrower gap between outliers and inliers, and a lot more overlap.
# This leads to more uncertain detection, even for more extreme outliers.
# Considering that different scores could be used differently in an operational
# environment, such as blocking all transactions above a certain threshold,
# while scores under that one but above another be put to manual
# inspection, we would like to minimize this gap where manual
# inspection is needed.
l2_scorer = OutlierDetectorScorer(test_Y, l2_scores)
print("\nl2 report:")
print(confusion_matrix(test_Y, l2_pred))
print(classification_report(test_Y, l2_pred))
print(f"AU-PRC:   {l2_scorer.auprc}")
print(f"baseline: {baseline}")
print(f"Threshold: {AE.l2_threshold}")
print(f"Optimal threshold: {l2_scorer.optimal_thresholds()[0]}")
prc_plot(l2_scorer.precisions, l2_scorer.recalls, l2_scorer.optimal_indices)
# AE-LL has stable performance, even when undersampling
# Log-Likelihood of the reconstruction errors provide better and more
# certain classifications. The overlapping section between outliers and inliers
# is noticeably narrowed, as outliers have more extreme scores.
# This is the better option with respect to the operational.
aell_scorer = OutlierDetectorScorer(test_Y, aell_scores)
print("\nAE-LL report:")
print(confusion_matrix(test_Y, aell_pred))
print(classification_report(test_Y, aell_pred))
print(f"AU-PRC:   {aell_scorer.auprc}")
print(f"baseline: {baseline}")
print(f"Threshold: {AE.LL.threshold}")
print(f"Optimal threshold: {aell_scorer.optimal_thresholds()[0]}")
prc_plot(aell_scorer.precisions, aell_scorer.recalls, aell_scorer.optimal_indices)

# Log-likelihood of the raw data gives comparable results
# to the auto-encoder recreation error log likelihood.
# This is probably because the inliers of the raw data
# are pretty evenly distributed, having low correlation,
# while the outliers have a significant correlation.
# This shows that a standard statistical model could be used
# for satisfactory results, as no advanced non-linear model
# is needed.
dll_scorer = OutlierDetectorScorer(test_Y, dll_scores)
print("\ndirect-LL report:")
print(confusion_matrix(test_Y, dll_pred))
print(classification_report(test_Y, dll_pred))
print(f"AU-PRC:   {dll_scorer.auprc}")
print(f"baseline: {baseline}")
print(f"Threshold: {direct_LL.threshold}")
print(f"Optimal threshold: {dll_scorer.optimal_thresholds()[0]}")
prc_plot(dll_scorer.precisions, dll_scorer.recalls, dll_scorer.optimal_indices)

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

if input("visualize with TSNE? y/n: ") == "y":
    # Visualize dataset with TSNE
    print("Visualizing dataset with TSNE")
    g = AE.scaler.transform(test_X)
    tsne = TSNE(
            n_components=2,
            verbose=2,
            n_iter=500,
            perplexity=10,
            random_state=1,
            learning_rate=200
            ).fit_transform(g)
    
    inlier_tsne, outlier_tsne = split_inliers_outliers(tsne, test_Y)
    inlier_tsne = np.asarray(inlier_tsne)
    outlier_tsne = np.asarray(outlier_tsne)
    
    # TSNE is unsupervised, and clusters most of the outliers in the
    # same cluster, indicating that outliers have distinguishing features.
    
    # Some outliers cluster together with inliers, indicating
    # that the difference between inliers and outliers is not trivial.
    scatterplot(
            (inlier_tsne, outlier_tsne), 
            alphas=(0.3, 0.5),
            cols=("yellow", "black"),
            labels=("inliers", "outliers"),
            title="TSNE - 2 components"
            )
    