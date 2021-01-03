# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 17:21:02 2020

@author: Ask
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import invgamma

def scatterplot(Xs, cols=None, alphas=None, labels=None, title=None):
    
    figure, ax = plt.subplots(figsize=(25, 25))
    scatters = []
    for i in range(len(Xs)):
        a = alphas[i] if alphas else 0.25
        c = cols[i] if cols else None
        s = ax.scatter(np.asarray(Xs[i][:,0]),np.asarray(Xs[i][:,1]), alpha=a, color=c)
        scatters.append(s)
    
    ax.legend(scatters, labels, scatterpoints=1, loc='lower left', fontsize=35)
    plt.title(title, fontsize=45)
    plt.xlabel('Z1', fontsize=30)
    plt.ylabel('Z2', fontsize=30)
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
    min_x = max(min_x, 1e-2)
    
    # Plot the training scores
    n_bins = 150
    if xscale == "log":
        bins = np.logspace(np.log10(min_x),np.log10(max_x),n_bins)
    else:
        bins = np.linspace(min_x, max_x, n_bins)

    fig = plt.figure(figsize=(15, 15))
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
    histogram(ax1, train_x,  "blue",   True,  "training data")
    histogram(ax2, inliers,  "yellow", True,  "inliers")
    histogram(ax3, outliers, "black",  False, "outliers")
    
    gamma = invgamma.pdf(bins, *p)
    
    # A lot of hacky stuff here
    
    # this curve is a bit bugged, adjusting the height fixes it
    ax4.set_ylim(0, max(gamma))
    ax4.plot(bins, gamma)
    
    [ax.set_xscale(xscale) for ax in (ax1, ax2, ax3, ax4)]
    
    # Ignore this, the library is being obstinate
    miny, maxy = plt.ylim()
    ax1.vlines(threshold, miny, maxy, color="black")
    ax2.vlines(threshold, miny, maxy, color="black")
    ax1.legend(("training data","threshold"), loc="upper right")
    ax2.legend(("inliers", "threshold"), loc="upper right")
    ax4.legend(["Gamma curve of best fit"], loc="center right")
    ax3.legend(["outliers"], loc="center right")
    plt.xlabel(xaxis)
    plt.ylabel("Density")
    plt.title(title)
    plt.show()


def prc_plot(precission, recall, optimal_indices):
    plt.plot(recall, precission)
    plt.scatter(recall[optimal_indices], precission[optimal_indices], color="red", label="optimal threshold")
    plt.legend(loc='upper right')
    plt.title("precission-recall curve")
    plt.xlabel("recall")
    plt.ylabel("precission")
    plt.show()