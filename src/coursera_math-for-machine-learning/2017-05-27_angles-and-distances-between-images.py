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


# 3. IMPORT MNIST DIGITS DATASET: ------------------------------------------
import sklearn
from sklearn.datasets import fetch_mldata
from ipywidgets import interact
MNIST = fetch_mldata('MNIST original', data_home='./MNIST')

plt.imshow(MNIST.data[MNIST.target==0].reshape(-1, 28, 28)[0], cmap='gray');
# todo: ^^ how does this work? 


'''
But we have the following questions:

1) What does it mean for two digits in the MNIST dataset to be different by our distance function?
2) Furthermore, how are different classes of digits different for MNIST digits? Let's find out!
For the first question, we can see just how the distance between digits compare among all distances for the first 500 digits;
'''
distances = []
for i in range(len(MNIST.data[:500])):
    for j in range(len(MNIST.data[:500])):
        distances.append(distance(MNIST.data[i], MNIST.data[j]))

len(distances)
distances[:50]

@interact(first=(0, 499), second=(0, 499), continuous_update=False)
def show_img(first, second):
    plt.figure(figsize=(8,4))
    f = MNIST.data[first].reshape(28, 28)
    s = MNIST.data[second].reshape(28, 28)
    
    ax0 = plt.subplot2grid((2, 2), (0, 0))
    ax1 = plt.subplot2grid((2, 2), (1, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    
    #plt.imshow(np.hstack([f,s]), cmap='gray')
    ax0.imshow(f, cmap='gray')
    ax1.imshow(s, cmap='gray')
    ax2.hist(np.array(distances), bins=50)
    d = distance(f, s)
    ax2.axvline(x=d, ymin=0, ymax=40000, color='C4', linewidth=4)
    ax2.text(0, 46000, "Distance is {:.2f}".format(d), size=12)
    ax2.set(xlabel='distance', ylabel='number of images')
    plt.show()



# 4. FIND MOST SIMILAR DIGIT TO THE FIRST DIGIT: ---------------------------
# let's try a toy exmaple: 
x = [1, 0, 4, -1]
min(x)
x.index(min(x))  # 3. Recall, python uses 0-based indexing


# now using the MNIST data: 
len(distances[1:500])  # exclude first element 
min(distances[1:500])

# find index of the min: 
distances[1:500].index(min(distances[1:500]))  # 60 

# this actually corresponds to 61 in the original data: 
distances[59:62]
# note: the way python slicing works, the above excludes item 62! 




# 5. FINDING MEAN IMAGE FOR EACH NUMERAL: ---------------------------------
# todo: how does all this work?? 
means = {}
for n in np.unique(MNIST.target).astype(np.int):
    means[n] = np.mean(MNIST.data[MNIST.target==n], axis=0)

MD = np.zeros((10, 10))
AG = np.zeros((10, 10))
for i in means.keys():
    for j in means.keys():
        MD[i, j] = distance(means[i], means[j])
        AG[i, j] = angle(means[i], means[j])

MD
AG


# visualize the distances: 
fig, ax = plt.subplots()
grid = ax.imshow(MD, interpolation='nearest')
ax.set(title='Distances between different classes of digits',
       xticks=range(10), 
       xlabel='class of digits',
       ylabel='class of digits',
       yticks=range(10))
fig.colorbar(grid)
plt.show()


# visualize the angles: 
fig, ax = plt.subplots()
grid = ax.imshow(AG, interpolation='nearest')
ax.set(title='Angles between different classes of digits',
       xticks=range(10), 
       xlabel='class of digits',
       ylabel='class of digits',
       yticks=range(10))
fig.colorbar(grid)
plt.show();





# 6. KNN ALGORITHM: -----------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
iris = datasets.load_iris()
print('x shape is {}'.format(iris.data.shape))
print('y shape is {}'.format(iris.target.shape))


# for simplicity use only first 2 dimensions: 
X = iris.data[:, :2] # use first two features for simplicity
y = iris.target


# visualize the data: 
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

fig, ax = plt.subplots(figsize=(4,4))

ax.scatter(X[:,0], X[:,1], c=y,
           cmap=cmap_bold, edgecolor='k', 
           s=20);
ax.legend()
ax.set(xlabel='$x_0$', ylabel='$x_1$', title='Iris flowers');



# define KNN function: 
def KNN(k, X, y, Xtest):
    """K nearest neighbors
    Arguments
    ---------
    k: int using k nearest neighbors.
    X: the training set
    y: the classes
    Xtest: the test set which we want to predict the classes

    Returns
    -------
    ypred: predicted classes for Xtest
    
    """
    N, D = X.shape
    M, D = Xtest.shape
    num_classes = len(np.unique(y))
    
    # 1. Compute distance with all flowers
    distance = pairwise_distance_matrix(X, Xtest) # EDIT THIS to use "pairwise_distance_matrix"

    # 2. Find indices for the k closest flowers
    idx = np.argsort(distance.T, axis=1)[:, :K]
    
    # 3. Vote for the major class
    ypred = np.zeros((M, num_classes))

    for m in range(M):
        klasses = y[idx[m]]    
        for k in np.unique(klasses):
            ypred[m, k] = len(klasses[klasses == k]) / K

    return np.argmax(ypred, axis=1)




    
# set K = 3 and display decision boundaries: 
K = 3

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
step = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))

ypred = []
data = np.array([xx.ravel(), yy.ravel()]).T
ypred = KNN(K, X, y, data)

fig, ax = plt.subplots(figsize=(4,4))

ax.pcolormesh(xx, yy, ypred.reshape(xx.shape), cmap=cmap_light)
ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20);
ax.set(xlabel='$x_0$', ylabel='$x_1$', title='KNN decision boundary (K={})'.format(K));















