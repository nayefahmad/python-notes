# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 22:59:27 2018

@author: Nayef
"""

#***************************************************************
# IMPLEMENTING PCA 
#***************************************************************
import numpy as np
import timeit

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


'''
Now we will implement PCA. Before we do that, let's pause for a moment and 
think about the steps for performing PCA. Assume that we are performing PCA 
on some dataset  XX  for  MM  principal components. We then need to perform 
the following steps, which we break into parts:

1. Data normalization (normalize).
2. Find eigenvalues and corresponding eigenvectors for the covariance matrix  SS .
 Sort by the largest eigenvalues and the corresponding eigenvectors (eig).

 After these steps, we can then compute the projection and reconstruction of 
the data onto the spaced spanned by the top  MM  eigenvectors
'''

# test array to be used for testing functions: 
test_array = np.array([[5,  2, -1],
                       [2,  2,  2],
                       [-1, 2,  5], 
                       [4, 2, 900]])
    
np.mean(test_array, axis = 0 )
#*************************************************************


# function to normalize dataset: 
def normalize(X):
    """Normalize the given dataset X
    Args:
        X: ndarray, dataset
    
    Returns:
        (Xbar, mean, std): ndarray, Xbar is the normalized dataset
        with mean 0 and standard deviation 1; mean and std are the 
        mean and standard deviation respectively.
    
    Note:
        You will encounter dimensions where the standard deviation is
        zero, for those when you do normalization the normalized data
        will be NaN. Handle this by setting using `std = 1` for those 
        dimensions when doing normalization.
    """
    mu = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    std_filled = std.copy()
    std_filled[std==0] = 1.
    Xbar = (X - mu)/std_filled
    return Xbar, mu, std

# test fn:
normalize(test_array)    



    
# function to compute eig values and eig vectors for the cov matrix S of 
#   the normalized data Xbar 
def eig(S):
    """Compute the eigenvalues and corresponding eigenvectors 
        for the covariance matrix S.
    Args:
        S: ndarray, covariance matrix
    
    Returns:
        (eigvals, eigvecs): ndarray, the eigenvalues and eigenvectors

    Note:
        the eigenvals and eigenvecs SHOULD BE sorted in descending
        order of the eigen values
        
        Hint: take a look at np.argsort for how to sort in numpy.
    """
    
    eigvals, eigvecs = np.linalg.eig(S)
    
    # index: first entry gives location of largest element: 
    i = np.argsort(-eigvals)
    
    # now resort: 
    eigvals = eigvals[i]
    eigvecs = eigvecs[i]
    
    return (eigvals, eigvecs) 

# test fn: 
eig(test_array[0:3, :])























    
    
    
    
    