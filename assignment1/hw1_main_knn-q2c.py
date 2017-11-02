"""
    Created on Nov 2, '17
    Person to send awards to if this script breaks loose, goes sentient and starts a new nuclear winter: Priyansh :]

    Same as hw1_main_knn but for running cross validation tests
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

    return y_preds, [np.mean(y_preds[key] == y_test) for key in _k]


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

# Make 5-fold cross validation splits.
indices = np.arange(1000)
logs = []
for i in range(5):
    print "\n\nRunning on split ", i+1
    np.random.shuffle(indices)
    test = indices[i*200:(i+1)*200]
    if i == 0:
        train = indices[(i+1)*200:]
    elif i == 4:
        train = indices[:i*200]
    else:
        train = np.concatenate((indices[:i*200], indices[(i+1)*200:]), axis=0)

    # Juxtapose indices on data
    X_train_local = X_train[train]
    X_test_local = X_train[test]
    y_train_local = y_train[train]
    y_test_local = y_train[test]
    k = range(1,16)

    # Run the experiment(s)
    y_pred, accuracies = run(X_train_local, y_train_local, X_test_local, y_test_local, _k=k)
    logs.append(accuracies)

logs = np.asarray(logs)
print logs
print np.sum(logs, axis=0)
print np.mean(logs, axis=0)

# Plotting time.
x = range(1,16)
for i in range(logs.shape[0]):
    plt.plot(x, logs[i], label='split '+str(i))
plt.plot(x, np.mean(logs, axis=0), label="avg", lw=2)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.legend(loc="upper right")
plt.show()










