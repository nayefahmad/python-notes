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

from sklearn.datasets import fetch_mldata
MNIST = fetch_mldata('MNIST original', data_home='./MNIST')
%matplotlib inline

plt.figure(figsize=(4,4))
plt.imshow(MNIST.data[0].reshape(28,28), cmap='gray');


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
    
    # now re-sort: 
    eigvals = eigvals[i]
    eigvecs = eigvecs[:, i]
    
    return (eigvals, eigvecs) 

# test fn: 
eig(test_array[0:3, :])




# function to get the projection matrix that projects onto the eigvecs of S: 
def projection_matrix(B):
    """Compute the projection matrix onto the space spanned by `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace
    
    Returns:
        P: the projection matrix
    """
    
    # Since we assume cols of B are an orthonormal basis, the proj 
    #   matrix has the simplified form B @ B-transpose 
    P = B @ np.linalg.inv(np.transpose(B) @ B) @ np.transpose(B)
    return P    




# function to actually do the PCA: 
def PCA(X, num_components):
    """
    Args:
        X: ndarray of size (N, D), where D is the dimension of the data,
           and N is the number of datapoints
        num_components: the number of principal components to use.
    Returns:
        X_reconstruct: ndarray of the reconstruction
        of X from the first `num_components` principal components.
    """
    # normalize the data: 
    # Xbar = normalize(X)[0]
    
    # Compute the data covariance matrix S
    S = np.cov(X)

    # Next find eigenvalues and corresponding eigenvectors for S by implementing eig().
    eig_vals, eig_vecs = eig(S)
    
    # Reconstruct the images from the lowerdimensional representation
    # To do this, we first need to find the projection_matrix (which you implemented earlier)
    # which projects our input data onto the vector space spanned by the eigenvectors
    P = projection_matrix(eig_vecs[:, :num_components]) # projection matrix
    
    # print(P)
    

    # Then for each data point x_i in the dataset X 
    #   we can project the original x_i onto the eigenbasis.
    X_reconstruct = P @ X
    return X_reconstruct

    
  

#*********************************************************************    
# IMPLEMENTING PCA ON MNIST DATASET: 
#*********************************************************************
    
## Some preprocessing of the data
NUM_DATAPOINTS = 1000
X = (MNIST.data.reshape(-1, 28 * 28)[:NUM_DATAPOINTS]) / 255.
Xbar, mu, std = normalize(X)

# testing: 
print(Xbar)
PCA(X, 1)


    
# define MSE: 
def mse(predict, actual):
    return np.square(predict - actual).sum(axis=1).mean()

    
    
    
'''
The greater number of of principal components we use, the smaller will our 
reconstruction error be. Now, let's answer the following question:

How many principal components do we need in order to reach a Mean Squared 
Error (MSE) of less than  100100  for our dataset?
'''
 
loss = []
reconstructions = []
for num_component in range(1, 100):
    reconst = PCA(Xbar, num_component)
    error = mse(reconst, Xbar)
    reconstructions.append(reconst)
    # print('n = {:d}, reconstruction_error = {:f}'.format(num_component, error))
    loss.append((num_component, error))

reconstructions = np.asarray(reconstructions)
reconstructions = reconstructions * std + mu # "unnormalize" the reconstructed image
loss = np.asarray(loss)    
    
print(loss)
 
   
# We can also put these numbers into perspective by plotting them.














    
    
    
    
    