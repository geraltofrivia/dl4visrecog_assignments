# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""
 
import numpy as np


def compute_euclidean_distances( X, Y ) :
    """
    Compute the Euclidean distance between two matricess X and Y
    Input:
    X: N-by-D numpy array
    Y: M-by-D numpy array

    Should return dist: M-by-N numpy array
    """

    # Find the dimensions of the space.
    op = np.asarray([[euclidean_distance(y, x) for x in X] for y in Y])
    return op


def euclidean_distance(x, y):
    """
    :param x_row: numpy 1D array
    :param y_row: numpy 1D array
    :return: numpy.float64 object
    """
    return np.sqrt(np.sum(np.power((x - y), 2)))

def predict_labels( dists, labels, k=1):
    """
    Given a Euclidean distance matrix and associated training labels predict a label for each test point.
    Input:
    dists: M-by-N numpy array 
    labels: is a N dimensional numpy array
    
    Should return  pred_labels: M dimensional numpy array

    Pseudocode:
        for a given row, sort descending. then take the majority vote from k.
    """

    op = []

    for row in dists:

        # Sort the indices of the array (ascending), and take k from the last
        indices = row.argsort()[:k]

        # Juxtapose these indices on labels to get k nearest labels
        nearest_labels = labels[indices]

        # Since its an unweighted knn, simply count the occurrences of each.
        votes = np.unique(nearest_labels, return_counts=True)

        # Select the most frequently occurring value
        winner = votes[0][np.argmax(votes[1])]

        # Pop goes the weasel
        op.append(winner)

    return np.asarray(op)
