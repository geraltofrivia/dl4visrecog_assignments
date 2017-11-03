"""
    Created on Nov 1, '17
    Person to send awards to if this script breaks loose, goes sentient and starts a new nuclear winter: Priyansh :]

"""

from load_mnist import *
import hw1_knn  as mlBasics
from sklearn.metrics import confusion_matrix
import numpy as np

# Load data - two class
# X_train, y_train = load_mnist('training' , [0,1] )
# X_test, y_test = load_mnist('testing'  ,  [0,1]  )

# Load data - ALL CLASSES
X_train, y_train = load_mnist('training')
X_test, y_test = load_mnist('testing')


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

    y_preds = {}  # Store ops in this

    # For all k,
    for k in _k:

        # Predict labels
        y_test_pred = mlBasics.predict_labels(dists, y_train, k=k)

        # Store them
        y_preds[k] = y_test_pred

    # Report results
    for key in y_preds:
        print '{0:0.02f}'.format(np.mean(y_preds[key] == y_test) * 100), "of test examples classified correctly. k =", key

    return y_preds


'''
    Create smaller subsets of the data.
    100 examples for each digit in training. total 1000 examples.
'''
# Sort the training data based on labels.
y_train_index = {}
for i in range(y_train.shape[0]):
    try:
        y_train_index[int(y_train[i])].append(i)
    except KeyError:
        y_train_index[int(y_train[i])] = [i]

# Sample 100 from each list (Assuming at least 100 examples of each sort
sample_indices = np.asarray([np.random.choice(y_train_index[i], size=100) for i in y_train_index], dtype=np.int).flatten()

# Using the index, recreate the X and Y train
X_train, Y_train = X_train[sample_indices], y_train[sample_indices]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# Run the experiment(s)
y_preds = run(X_train, y_train, X_test, y_test, _k=[1, 5])

# 4) Visualize Nearest Neighbors
# @TODO: How the fuck

# 5) Confusion Matrix
print 'Confusion matrix for k=1'
print confusion_matrix(y_test, y_preds[1])
print 'Confusion matrix for k=5'
print confusion_matrix(y_test, y_preds[5])


