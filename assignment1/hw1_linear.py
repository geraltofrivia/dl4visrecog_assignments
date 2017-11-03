# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""

import numpy as np 
 

def predict(X, W, b):
    """
    implement the function h(x, W, b) here  
    X: N-by-D array of training data 
    W: D dimensional array of weights
    b: scalar bias

    Should return a N dimensional array  
    """
    U = np.dot(X, W) + b
    V = sigmoid(U)
    return V

 
def sigmoid(a): 
    """
    implement the sigmoid here
    a: N dimensional numpy array

    Should return a N dimensional array  
    """
    return 1/(1+ np.exp(-a))


def l2loss(X, y, W, b):
    """
    implement the L2 loss function
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias

    Should return three variables: (i) the l2 loss: scalar, (ii) the gradient with respect to W, (iii) the gradient with respect to b
    """
    v = predict(X, W, b)
    l2 = np.sum(np.square(y-v))

    de = y-v
    dv = de*(v*(1-v))
    dv = dv.reshape((dv.shape[0], 1))

    dw = np.mean((-2*X)*dv, axis=0)
    db = np.mean(-2*dv, axis=0)
    # @TODO: Verify dimensions

    return l2, dw, db


def train(X,y,W,b, num_iters=1000, eta=0.001):  
    """
    implement the gradient descent here
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias
    num_iters: (optional) number of steps to take when optimizing
    eta: (optional)  the stepsize for the gradient descent

    Should return the final values of W and b    
    """
    log = []

    for i in range(num_iters):

        # Run the data with current params
        l, dw, db = l2loss(X, y, W, b)

        # Update params
        W -= dw
        b -= db

        log.append(l)

    return W, b, log 

