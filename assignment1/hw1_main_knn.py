"""
    Created on Nov 1, '17
    Person to send awards to if this script breaks loose, goes sentient and starts a new nuclear winter: Priyansh :]

"""

from load_mnist import *
import hw1_knn  as mlBasics
import numpy as np

# Load data - two class
# X_train, y_train = load_mnist('training' , [0,1] )
# X_test, y_test = load_mnist('testing'  ,  [0,1]  )

# Load data - ALL CLASSES
X_train, y_train = load_mnist('training')
X_test, y_test = load_mnist('testing')

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

# Test on test data
# 1) Compute distances:
dists = mlBasics.compute_euclidean_distances(X_train, X_test)

# 2) Run the code below and predict labels:
giy_test_pred_k1 = mlBasics.predict_labels(dists, y_train, k=1)
y_test_pred_k5 = mlBasics.predict_labels(dists, y_train, k=5)

# Visualizing

# 3) Report results
# you should get following message '99.91 of test examples classified correctly.'
print '{0:0.02f}'.format(np.mean(y_test_pred_k1 == y_test) * 100), "of test examples classified correctly. k=1"
print '{0:0.02f}'.format(np.mean(y_test_pred_k5 == y_test) * 100), "of test examples classified correctly. k=5"


