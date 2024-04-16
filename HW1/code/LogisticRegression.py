import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

		### YOUR CODE HERE
        self.W = np.zeros(n_features) 

        for epoch in range(self.max_iter):
            # Calculate predictions
            predictions = self.predict_proba(X)

            # Compute gradient
            gradient = -X.T @ (y - predictions[:, 1]) / n_samples

            # Update weights
            self.W -= self.learning_rate * gradient


		### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)

        for epoch in range(self.max_iter):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]

            for i in range(0, n_samples, batch_size):
                # Select a mini-batch
                X_mini_batch, y_mini_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]

                # Calculate predictions
                predictions = self.predict_proba(X_mini_batch)

                # Compute gradient for the mini-batch
                gradient = -X_mini_batch.T @ (y_mini_batch - predictions[:, 1]) / len(X_mini_batch)

                # Update weights
                self.W -= self.learning_rate * gradient



		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE

        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)

        for epoch in range(self.max_iter):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]

            for i in range(n_samples):
                # Select one sample
                xi, yi = X_shuffled[i], y_shuffled[i]

                # Calculate prediction
                prediction = self.predict_proba(xi)

                # Compute gradient for the current sample
                gradient = -xi * (yi - prediction[:, 1])

                # Update weights
                self.W -= self.learning_rate * gradient


		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        
        # Compute the dot product of _w and _x
        _wx = np.dot(self.W, _x) 

         # Compute the sigmoid function value
        sigmoid_wx = 1 / (1 + np.exp(-_wx*_y))

        # Compute the gradient using the derived formula 
        _g = _y*_x*np.exp(-_wx*_y) * sigmoid_wx
        #_g = -(_y / sigmoid_wx - (1 - _y) / (1 - sigmoid_wx)) * _x * sigmoid_wx * (1 - sigmoid_wx)

        return _g 


		### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE

        self.get_params()

        # Calculate the dot product of X and W
        wx = X @ self.W

        # Calculate the sigmoid function values for class 1 and 0
        sigmoid_wx = 1 / (1 + np.exp(-wx))
        preds_proba = np.column_stack((1 - sigmoid_wx, sigmoid_wx))

        return preds_proba

		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE

        self.get_params()

        # Calculate the dot product of X and W
        wx = X @ self.W

        # Apply the sign function to get the predicted class labels
        preds = np.sign(wx)

        return preds 

		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE

        self.get_params()

        # Make predictions
        predictions = self.predict(X)

        # Calculate accuracy
        accuracy = np.mean(predictions == y)

        return accuracy 

		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

