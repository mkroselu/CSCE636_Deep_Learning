import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"


# code used for visualize_features 
train_data, train_labels = load_data(os.path.join(data_dir, train_filename))
train_X = np.array([prepare_X(raw_X) for raw_X in train_data]) # training data with 3 features 
train_X_2 = train_X[ :, 1:] # 2 Features without bias
train_y, idx = prepare_y(train_labels) 

visualize_features(train_X_2, train_y) 


# code used for visualize_result 
train_data, train_labels = load_data(os.path.join(data_dir, train_filename))
train_X = np.array([prepare_X(raw_X) for raw_X in train_data]) # training data with 3 features 
train_y, idx = prepare_y(train_labels)  
train_X = train_X[idx]

train_X_2 = train_X[ :, 1:] # 2 Features without bias 
train_y = train_y[(train_y == 1) | (train_y == 2)]
train_y[(train_y == 2)] = -1 

bgd_model = logistic_regression(learning_rate = 0.01, max_iter=100000)
bgd_model.fit_BGD(train_X, train_y)

# Access the trained weights
trained_weights = bgd_model.W

# Print the weights
print("Trained Weights:", trained_weights)

visualize_result(train_X_2, train_y, trained_weights) 


# code used for visualize_result_multi 
train_data, train_labels = load_data(os.path.join(data_dir, train_filename))
train_X = np.array([prepare_X(raw_X) for raw_X in train_data]) # training data with 3 features 
train_y, idx = prepare_y(train_labels)  
train_X_2 = train_X[ :, 1:] # 2 Features without bias 

visualize_result_multi(train_X_2, train_y, W)


def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
	
    # Filter samples for class 1 and 2
    class1 = X[y == 1]
    class2 = X[y == 2]

    # Scatter plot
    plt.scatter(class1[:, 0], class1[:, 1], label='Class 1', marker='o')
    plt.scatter(class2[:, 0], class2[:, 1], label='Class 2', marker='x')

    # Set plot labels and title
    plt.xlabel('Symmetry')
    plt.ylabel('Intensity')
    plt.title('Visualization of Training Features for Class 1 and 2')

    # Add legend
    plt.legend()

    # Save the plot
    plt.savefig('train_features.png')

    # Show the plot
    plt.show()

    ### END YOUR CODE

def visualize_result(X, y, W):
	'''This function is used to plot the sigmoid model after training. 

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 1 or -1.
		W: An array of shape [n_features,].
	
	Returns:
		No return. Save the plot to 'train_result_sigmoid.*' and include it
		in submission.
	'''
	### YOUR CODE HERE



    # Plotting the decision boundary
    #x1 = np.linspace(min(X[train_y ==1][:, 0]), max(X[train_y == -1][:, 0]), 100)
    x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x2 = -(W[0] + W[1] * x1) / W[2] 

    #plt.figure(figsize=(8, 6))
    
    # Plot the data points
    
    # Filter samples for class 1 and 2
    class1 = X[train_y == 1]
    class2 = X[train_y == -1]

    # Scatter plot
    plt.scatter(class1[:, 0], class1[:, 1], label='Class 1', marker='o')
    plt.scatter(class2[:, 0], class2[:, 1], label='Class 2', marker='x')
    
    #plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Paired, edgecolors = 'k', marker = 'o', s = 50)
    
    # Plot the decision boundary
    plt.plot(x1, x2, color = 'blue', linewidth = 2)

    plt.title('Decision Boundary of mini BGD')
    plt.xlabel('Symmetry')
    plt.ylabel('Intensity')
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig('train_result_sigmoid_miniBGD.png')
    
    # Display the plot
    plt.show()

	### END YOUR CODE

def visualize_result_multi(X, y, W):
	'''This function is used to plot the softmax model after training. 

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 0,1,2.
		W: An array of shape [n_features, 3].
	
	Returns:
		No return. Save the plot to 'train_result_softmax.*' and include it
		in submission.
	'''
	 ### YOUR CODE HERE

    # Plotting the decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = np.c_[xx.ravel(), yy.ravel()] @ W.T 
    Z = np.argmax(Z, axis=0)

    # Reshape Z to the shape of xx only if it's not a scalar
    if np.isscalar(Z):
        Z = np.full_like(xx, Z)
    else:
        Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Scatter plot for different classes
    for i in range(3):
        plt.scatter(X[y == i][:, 0], X[y == i][:, 1], label=f'Class {i}', marker='o')

    plt.title('Decision Boundaries of Softmax Model')
    plt.xlabel('Symmetry')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig('train_result_softmax.png')

    # Display the plot
    plt.show()

	### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0] 

#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE

    logisticR_classifier = logistic_regression(learning_rate=0.01, max_iter=100)

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    logisticR_classifier = logistic_regression(learning_rate=0.01, max_iter=1000)

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    ### END YOUR CODE

	# Visualize the your 'best' model after training.
	# visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

    ### YOUR CODE HERE

    train_data, train_labels = load_data(os.path.join(data_dir, train_filename))
    train_X = np.array([prepare_X(raw_X) for raw_X in train_data]) # training data with 3 features 
    train_y, idx = prepare_y(train_labels)  
    train_X = train_X[idx]

    train_X_2 = train_X[ :, 1:] # 2 Features without bias 
    train_y = train_y[(train_y == 1) | (train_y == 2)]
    train_y[(train_y == 2)] = -1 

    sgd_model = logistic_regression(learning_rate=0.05, max_iter=1000)
    sgd_model.fit_SGD(train_X, train_y) 

    # Access the trained weights
    trained_weights = sgd_model.W

    # Print the weights
    print("Trained Weights:", trained_weights)

    print(sgd_model.get_params())
    print(sgd_model.score(train_X, train_y))

    visualize_result(train_X_2, train_y, trained_weights)


    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE

    # Testing process 
    # code used for visualize_result 
    train_data, train_labels = load_data(os.path.join(data_dir, test_filename))
    train_X = np.array([prepare_X(raw_X) for raw_X in train_data]) # training data with 3 features 
    train_y, idx = prepare_y(train_labels)  
    train_X = train_X[idx]

    train_X_2 = train_X[ :, 1:] # 2 Features without bias 

    sgd_model = logistic_regression(learning_rate=0.05, max_iter=1000)
    sgd_model.fit_SGD(train_X, train_y) 

    # Access the trained weights
    trained_weights = sgd_model.W

    # Print the weights
    print("Trained Weights:", trained_weights)

    print(sgd_model.get_params())
    print(sgd_model.score(train_X, train_y))

    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)


    # Explore different hyper-parameters.
    ### YOUR CODE HERE

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.01, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)


    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.01, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 100)


    ### END YOUR CODE

	# Visualize the your 'best' model after training.
	# visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())

    train_data, train_labels = load_data(os.path.join(data_dir, train_filename))
    train_X = np.array([prepare_X(raw_X) for raw_X in train_data]) # training data with 3 features 
    train_y, idx = prepare_y(train_labels)  
    train_X_2 = train_X[ :, 1:] # 2 Features without bias 

    visualize_result_multi(train_X_2, train_y, W) 

    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE

    train_data, train_labels = load_data(os.path.join(data_dir, test_filename))
    train_X = np.array([prepare_X(raw_X) for raw_X in train_data]) # training data with 3 features 
    train_y, idx = prepare_y(train_labels)  
    train_X_2 = train_X[ :, 1:] # 2 Features without bias 

    visualize_result_multi(train_X_2, train_y, W)

    ### END YOUR CODE


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE



    ### END YOUR CODE


    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1   

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE



    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


'''
Explore the training of these two classifiers and monitor the graidents/weights for each step. 
Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
Then, for what learning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
'''
    ### YOUR CODE HERE



    ### END YOUR CODE

    # ------------End------------
    

if __name__ == '__main__':
	main()
    
    
