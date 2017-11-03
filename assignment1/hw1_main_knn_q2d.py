"""
    Created on Nov 2, '17
    Person to send awards to if this script breaks loose, goes sentient and starts a new nuclear winter: Priyansh :]

    Instead of using 100 examples per digit class, use all images in the training set (60000 samples).
    Report your observations for k = 1 neighbor and the k neighbors which performed best in your cross validation evaluation.
    How much did your functions' computational cost of classifying the test data increased?
    How well did your classifier perform on the test set?
"""

from load_mnist import *
import hw1_knn  as mlBasics
import numpy as np
from matplotlib import pyplot as plt

# Load data - two class
# X_train, y_train = load_mnist('training' , [0,1] )
# X_test, y_test = load_mnist('testing'  ,  [0,1]  )

# Load data - ALL CLASSES
X_train, y_train = load_mnist('training')
X_test, y_test = load_mnist('testing')

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

def run(X_train, y_train, X_test, y_test, _k=[1]):
    """
    Script to run the experiment given some data. It would train the Knn (compute n x n distances).
    And then predict labels for the test set.

    :param X_train: np mat, dimensions: N x D
    :param y_train: np mat, dimensions: N
    :param X_test: np mat, dimensions: M x D
    :param y_test: np mat, dimensions: M
    :param _k: list of int. How many k's to test for.

    :return: y_pred: np mat, dimensions: M
    """
    # Compute distances:
    dists = mlBasics.compute_euclidean_distances(X_train, X_test)

    # y_preds = {}  # Store ops in this

    # For all k,
    for k in _k:

        # Predict labels
        y_test_pred = mlBasics.predict_labels(dists, y_train, k=k)

        print '{0:0.02f}'.format(np.mean(y_test_pred == y_test) * 100), "of test examples classified correctly. k =", key
        # Store them
        # y_preds[k] = y_test_pred


    # return y_preds, [np.mean(y_preds[key] == y_test) for key in _k]

# Run the experiment
run(X_train, y_train, X_test, y_test, _k=[1, 15])













