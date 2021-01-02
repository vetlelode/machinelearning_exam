# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 20:48:24 2020

@author: Ask
"""

# This file is unfinished. It is only here to show that we attempted to write
# The regressor ourselves.
import numpy as np
import random

def TotalSumOfSquares( Y, Y_bar ):
    return np.square(Y-Y_bar).sum()

def MeanSquareError( Y, Y_bar ):
    Y, Y_bar = np.array(Y), np.array(Y_bar)
    return TotalSumOfSquares(Y, Y_bar)/len(Y)

def R2( Y, Y_bar ):
    Y, Y_bar =np.asarray(Y), np.asarray(Y_bar)
    mean = np.mean(Y)
    return 1-(TotalSumOfSquares(Y, Y_bar)/TotalSumOfSquares(Y, mean))

def average_weight( W ):
    #list of weights
    return [np.mean(w) for w in W]

# UNFINISHED. DOES NOT WORK.
class CustomRegressor:
    def __init__(
            self,
            layer_sizes,
            activation = "sigmoid",
            xnoise = 0,
            ynoise = 0,
            epochs = 50,
            cost = lambda y, yb: y-yb,
            batchsize=64,
            learningrate=0.1
            ):
        self.weights = [np.random.randn(a,b) for a,b in zip(layer_sizes[1:],layer_sizes[:-1])]
        self.biases = [np.random.randn(a,1) for a in layer_sizes[1:]]
        self.n_of_layers = len(layer_sizes)
        
        S = lambda z: np.exp(z)/(1+np.exp(z))
        activations = {
                "elu":(
                        lambda z: z if z>0 else np.exp(z)-1,
                        lambda z: z if z>0 else np.exp(z)
                        ),
                "relu":(
                        lambda z: z if z>0 else 0,
                        lambda z: 1 if z>0 else 0
                        ),
                "tanh":(
                        lambda z: (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z)),
                        lambda z: (4*np.exp(2*z))/((np.exp(2*z)+1)**2)
                        ),
                "sigmoid":(
                        S,
                        lambda z: S(z)*(1-S(z))
                        )
                }
        self.activation = activations[activation][0]
        self.derivative_activation = np.vectorize(activations[activation][1])
        self.xnoise = xnoise
        self.ynoise = ynoise
        self.epochs = epochs
        self.cost = cost
        self.batchsize = batchsize
        self.learningrate = learningrate
    
    def fit( self, X, Y ):
        n = len(X)
        XY = list(zip(X,Y))
        batchsize = min(self.batchsize, n)
        addnoise = lambda m,noise: (np.random.rand(*m.shape)*noise) + m
        for _ in range(self.epochs):
            random.shuffle(XY)
            for i in range(0, n, batchsize):
                batch_XY = np.asmatrix(XY[i:i+batchsize])
                batch_X = batch_XY[:,0]
                batch_Y = batch_XY[:,1]
                #add noise
                
                batch_Y = addnoise(batch_Y,self.xnoise)
                batch_X = addnoise(batch_X,self.ynoise)
                
                Z, A = self.encode(batch_X)
                dW, dB = self.back_propogate(batch_X, Z, A)
                
                self.weights = [weight+self.learningrate*dw for weight,dw in zip(self.weights, dW)]
                self.biases = [bias+self.learningrate*db for bias,db in zip(self.biases, dB)]
                print("loss = {}".format(np.linalg.norm(self.cost(batch_Y,activations[-1]) )))
                
    @staticmethod
    def encode(network, X):
        z = np.asmatrix(X)
        a = z
        Z = []
        A = [a]
        for weight, bias, activation in network:
            # concatenate the bias to fit the weight-layer multiplication
            # sum the weights and then add the bias
            z = np.add(np.matmul(z, weight), bias)
            a = activation(z)
            Z.append(z)
            A.append(a)
        return Z, A
    
    
    def forward_propogate(self, X):
        return CustomRegressor.encode(self.network, X)
    
    #https://medium.com/@a.mirzaei69/implement-a-neural-network-from-scratch-with-python-numpy-backpropagation-e82b70caa9bb
    def back_propogate( self, Y, Z, A ):
        
        
        error = (expected - output) * transfer_derivative(output)
        
        dW = []
        dB = []
        delta = self.cost(Y, A[-1]*self.network[-1][-1](Z[-1]))
        deltas = []
        for weight, bias, activation, activation_prime in reversed(self.network):
            delta = np.matmul(weight,delta) * activation
        
        for i in reversed(range(len(deltas)-1)):
            deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*(self.activation(zs[i]))
            batch_size = Y.shape[1]
            dB = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas]
            dW = [d.dot(activations[i].T)/float(batch_size) for i,d in enumerate(deltas)]
        
        return dW, dB

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


import matplotlib.pyplot as plt
nn = CustomRegressor([1, 100, 1])
X = 2*np.pi*np.random.rand(1000).reshape(1, -1)
y = np.sin(X)

nn.fit(X, y)
_, a_s = nn.feedforward(X)
#print(y, X)
plt.scatter(X.flatten(), y.flatten())
plt.scatter(X.flatten(), a_s[-1].flatten())
plt.show()