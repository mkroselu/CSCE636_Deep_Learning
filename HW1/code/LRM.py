#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: 
"""

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)  # Initialize weights as zeros

        # Convert labels to one-hot encoding
        one_hot_labels = np.eye(self.k)[labels.astype(int)]

        for epoch in range(self.max_iter):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], one_hot_labels[indices]

            for i in range(0, n_samples, batch_size):
                # Select a mini-batch
                X_mini_batch, y_mini_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]

                # Compute gradient for the mini-batch
                gradient = self._gradient(X_mini_batch, y_mini_batch)

                # Update weights
                self.W -= self.learning_rate * gradient

        return self


		### END YOUR CODE
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features, k]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE

        # Compute the dot product of _w and _x
        _wx = np.dot(_x, self.W)

        # Compute the softmax function values
        softmax_values = self.softmax(_wx)

        # Compute the gradient using the derived formula 
        _g = -np.outer(_x.T, (_y - softmax_values)) / len(_x)

        return _g 

		### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		### YOUR CODE HERE
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
		### END YOUR CODE
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features, k].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE

        logits = np.dot(X, self.W)
        probabilities = self.softmax(logits)
        preds = np.argmax(probabilities, axis=1)
        
        return preds

		### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		### YOUR CODE HERE

        preds = self.predict(X)
        accuracy = np.mean(preds == labels)
        
        return accuracy

		### END YOUR CODE

