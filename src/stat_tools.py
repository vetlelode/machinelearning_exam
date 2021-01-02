# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 17:09:27 2020

@author: Ask
"""

import numpy as np
from scipy.stats import invgamma


def split_inliers_outliers(X, Y):
    inliers, outliers = [], []
    for i in range(len(X)):
        if Y[i] == 0:
            inliers.append(X[i])
        else:
            outliers.append(X[i])
    return np.asarray(inliers), np.asarray(outliers)

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
        #self.lnσs = np.log(self.σ2s) *0.5
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
        # only that the order of scores are the same.
        
        # f(x)=e^(-(x - μ)^2/(2 σ^2))/(sqrt(2 π) σ)
        # g(x)=(x - μ)^2 / (σ^2). The ln(σ) can be dropped too
        
        # ln(f(x))=-(x - μ)^2/(2 σ^2)-ln(sqrt(2 π) σ)
        # ln(f(x))=-(x - μ)^2/(σ^2) /2-ln(sqrt(2 π)) -ln(σ)
        # ln(f(x))*2+ln(sqrt(2 π))=-(x - μ)^2/(σ^2) -ln(σ)
        # ln(f(x))*2+ln(sqrt(2 π))+ln(σ)=-(x - μ)^2/(σ^2)

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
