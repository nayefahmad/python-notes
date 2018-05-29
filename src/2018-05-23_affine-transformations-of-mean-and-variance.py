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


# --------------------------------------------------------------------------------
# 1. Olivetti Faces dataset: ------------------------------
# --------------------------------------------------------------------------------

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

# faces.shape

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
    X: (N, D) ndarray representing the dataset. N rows, D columns. 
    
    Returns
    -------
    mean: (D, ) ndarray which is the mean of the dataset.
    """
    N, D = X.shape
    mean = np.zeros(D)
    for n in range(N):
        mean_row = np.mean(X[n]) 
        mean.append(mean_row)
    return mean
    
# test the function: 
X = random.randn(10, 4); X
# e.g. height, weight, IQ and salary for 10 ppl 

mean_naive(X)  # scalar value 4 # todo: is this right? 




def cov_naive(X):
    """Compute the covariance for a dataset
    Arguments
    ---------
    X: (N, D) ndarray representing the dataset.
    
    Returns
    -------
    covariance: (D, D) ndarray which is the covariance matrix of the dataset.
    
    """
    N, D = X.shape
    covariance = np.zeros((D, D))
    for n in range(N):
        covariance = np.cov(X) 
    return covariance    


# test the function: 
X = random.randn(10, 4); X
# e.g. height, weight, IQ and salary for 10 ppl 

cov_naive(X)
cov_naive(X).shape  # N by N matrix



# --------------------------------------------------------------------------------
# 3. MEAN AND COVARIANCE USING VECTORIZED OPERATIONS: ------------------------
# --------------------------------------------------------------------------------

def mean(X):
    """Compute the mean for a dataset
    
    Arguments
    ---------
    X: (N, D) ndarray representing the dataset.
    
    Returns
    -------
    mean: (D, ) ndarray which is the mean of the dataset.
    """
    mean = X.mean(axis=0) 
    return mean


def cov(X):
    """Compute the covariance for a dataset
    Arguments
    ---------
    X: (N, D) ndarray representing the dataset.
    
    Returns
    -------
    covariance_matrix: (D, D) ndarray which is the covariance matrix of the dataset.
    
    """
    # It is possible to vectorize our code for computing the covariance, i.e. we do not need to explicitly
    #    iterate over the entire dataset as looping in Python tends to be slow
    N, D = X.shape
    covariance_matrix = np.cov(X) 
    return covariance_matrix



# calculate the mean face: 
def mean_face(faces):
    """Compute the mean of the `faces`
    
    Arguments
    ---------
    faces: (N, 64 * 64) ndarray representing the faces dataset.
    
    Returns
    -------
    mean_face: (64, 64) ndarray which is the mean of the faces.
    """
    mean_face = mean(faces)
    return mean_face

plt.imshow(mean_face(faces).reshape((64, 64)), cmap='gray');



# --------------------------------------------------------------------------------
# 4. PERFORMANCE BENCHMARKING WITH TIME FUNCTIONS: -------------------------------------------
# --------------------------------------------------------------------------------

# We have some huge data matrix, and we want to compute its mean
X = np.random.randn(100000, 20)
# Benchmarking time for computing mean
%time mean_naive(X)
%time mean(X)
pass


'''
Alternatively, we can also see how running time increases as we increase the 
size of our dataset. In the following cell, we run mean, mean_naive and cov, 
cov_naive for many times on different sizes of the dataset and collect their
running time. If you are less familiar with Python, you may want to spend 
some time understanding what the code does. 
''''
 
# define function to measure time: 
def time(f, repeat=100):
    """A helper function to time the execution of a function.
    
    Arguments
    ---------
    f: a function which we want to time it.
    repeat: the number of times we want to execute `f`
    
    Returns
    -------
    the mean and standard deviation of the execution.
    """
    times = []  # initialize empty list to collect durations 
    
    for _ in range(repeat):  # todo: why use underscore here instead of character like i? 
        start = timeit.default_timer()
        f()  # call the function 
        stop = timeit.default_timer()
        times.append(stop-start)  # calculate duration 
    
    return np.mean(times), np.std(times)




# now measure difference between naive and vectorized mean: 
fast_time = []
slow_time = []

for size in np.arange(100, 300, step=100):  # todo: what is np.arrange()  ? 
    X = np.random.randn(size, 20)
    
    f = lambda : mean(X)  # first pass the vectorized mean fn to time( )
    mu, sigma = time(f)
    fast_time.append((size, mu, sigma))
    
    f = lambda : mean_naive(X)  # then pass the naive mean to time( ) 
    mu, sigma = time(f)
    slow_time.append((size, mu, sigma))

fast_time = np.array(fast_time)
slow_time = np.array(slow_time)    

fast_time
slow_time


# plot the change in time with change in size of input using matplotlib: 
fig, ax = plt.subplots()

ax.errorbar(fast_time[:,0], fast_time[:,1], fast_time[:,2], label='fast mean', linewidth=2)
ax.errorbar(slow_time[:,0], slow_time[:,1], slow_time[:,2], label='naive mean', linewidth=2)

ax.set_xlabel('size of dataset')
ax.set_ylabel('running time')
plt.legend(); 



# --------------------------------------------------------------------------------
# 5. AFFINE TRANSFORMATION OF DATA: -------------------------------------------
# --------------------------------------------------------------------------------
def affine_mean(mean, A, b):
    """Compute the mean after affine transformation
    Args:
        mean: ndarray, the mean vector
        A, b: affine transformation applied to x
    Returns:
        mean vector after affine transformation
    """
    affine_m = (A @ mean) + b 
    return affine_m


def affine_covariance(S, A, b):
    """Compute the covariance matrix after affine transformation
    Args:
        S: ndarray, the covariance matrix
        A, b: affine transformation applied to each element in X        
    Returns:
        covariance matrix after the transformation
    """
    affine_cov = A @ (S @ np.transpose(A)) 
    return affine_cov



# generate test data: ------------------------------------------------------ 
random = np.random.RandomState(42)  # set seed for random num
# random

A = random.randn(4,4); A
b = random.randn(4); b; b.shape  # vector of 4 elements

# create dataset: 100 observations of 4 variables: 
X = random.randn(100, 4); X
X.shape  # (100, 4)

mean(X); 
mean(X).shape  # (4, ) ==> vector of 4 elements

X1 = ((A @ (X.T)).T + b)  # applying affine transformation once
X2 = ((A @ (X1.T)).T + b) # and again

# --------------------------------------------------------------------------


'''
One very useful way to compare whether arrays are equal/similar is use the 
helper functions in numpy.testing. the functions in numpy.testing will throw 
an AssertionError when the output does not satisfy the assertion.
'''

# check mean function: -----------------------------------------------------
np.testing.assert_almost_equal(mean(X1),
                               affine_mean(mean(X), A, b))
print('correct')



# check covariance function: ----------------------------------------------------- 
np.testing.assert_almost_equal(cov(X1),  
                               affine_covariance(cov(X), A, b))
print('correct')  # todo: why doesn't this work? 

affine_covariance(cov(X), A, b)
# shapes (2,2) and (4,4) not aligned: 2 (dim 1) != 4 (dim 0)

cov(X).shape  # (2,2)





































