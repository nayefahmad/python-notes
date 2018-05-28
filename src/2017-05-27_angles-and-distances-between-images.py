# -*- coding: utf-8 -*-
"""
Created on Sun May 27 18:09:34 2018

@author: Nayef
"""

#************************************************************
# DISTANCES AND ANGLES BETWEEN IMAGES 
#************************************************************

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy


# 1. DEFINE FUNCTION: ------------------------------------------------------
def distance(x, y):
    """Compute distance between two vectors x, y using the dot product"""
    x = np.array(x, dtype=np.float).ravel() # ravel() "flattens" the ndarray
    y = np.array(y, dtype=np.float).ravel()
    distance = np.sqrt((x-y) @ (x-y))
    return distance


    
def angle(x, y):
    """Compute the angle between two vectors x, y using the dot product"""
        
    angle = np.arccos(x@y/(np.sqrt(x@x)*np.sqrt(y@y)))
    return angle


def pairwise_distance_matrix(X, Y):
    """Compute the pairwise distance between rows of X and rows of Y

    Arguments
    ----------
    X: ndarray of size (N, D)
    Y: ndarray of size (M, D)
    
    Returns
    --------
    D: matrix of shape (N, M), each entry D[i,j] is the distance between
    X[i,:] and Y[j,:] using the dot product.
    """
    N, D = X.shape
    M, D = Y.shape
    
    import sklearn.metrics.pairwise
    distance_matrix = sklearn.metrics.pairwise.pairwise_distances(X, Y)
        
    return distance_matrix


# Testing the function: 
M = np.array([[2, -1], 
              [-1, 4]])    
    
N = np.array([[2, -1], 
              [-1, 4] ,
              [4,10]])    
    
pairwise_distance_matrix(M, N)    


# 2. PLOTTING AND TESTING: ------------------------------------------------------

# function for plotting: 
def plot_vector(v, w):
    """Plot two vectors `v` and `w` of dimension 2"""
    fig = plt.figure(figsize=(4,4))
    ax = fig.gca()
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.grid()
    ax.arrow(0, 0, v[0], v[1], head_width=0.05, head_length=0.1, 
             length_includes_head=True, linewidth=2, color='r');
    ax.arrow(0, 0, w[0], w[1], head_width=0.05, head_length=0.1, 
             length_includes_head=True, linewidth=2, color='r');
             
# Some sanity checks, you may want to have more interesting test cases to test your implementation
a = np.array([1,0])
b = np.array([0,1])
np.testing.assert_almost_equal(distance(a, b), np.sqrt(2))
assert((angle(a,b) / (np.pi * 2) * 360.) == 90)
print('correct')

plot_vector(b, a)

























