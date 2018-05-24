# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:28:00 2018

@author: nahmad3
"""

#************************************************************************
# EFFECT OF AFFINE TRANSFORMATIONS ON MEAN AND VARIANCE 
#************************************************************************

# import packages: 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.datasets import fetch_lfw_people, fetch_mldata, fetch_olivetti_faces
import time
import timeit

%matplotlib inline
from ipywidgets import interact



# 1. Olivetti Faces dataset: ------------------------------
'''
The dataset we have are usually stored as 2D matrices, then it would be really 
important to know which dimension represents the dimension of the dataset, and 
which represents the data points in the dataset.
'''

image_shape = (64, 64)
# Load faces data
dataset = fetch_olivetti_faces()
faces = dataset.data

print('Shape of the faces dataset: {}'.format(faces.shape))
# note: 64 * 64 = 4096
# i.e., the 2d image is stored as a single vector 

print('{} data points'.format(faces.shape[0]))
# there are 400 vectors in our dataset 


# examine the images with interact widget: ---------------
# This works well only in Jupyter notebooks 
@interact(n=(0, len(faces)-1))
def display_faces(n=0):
    plt.figure()
    plt.imshow(faces[n].reshape((64, 64)), cmap='gray')
    plt.show()






# 2. MEAN AND COVARIANCE OF THE DATA: NAIVE METHOD: ------------------------
# Naive method uses iteration, rather than vectorized operations 
    
def mean_naive(X):
    """Compute the mean for a dataset by iterating over the dataset
    
    Arguments
    ---------
    X: (N, D) ndarray representing the dataset.
    
    Returns
    -------
    mean: (D, ) ndarray which is the mean of the dataset.
    """
    N, D = X.shape
    mean = np.zeros(D)
    for n in range(N):
        mean = np.mean(D) # EDIT THIS
    return mean
    
    