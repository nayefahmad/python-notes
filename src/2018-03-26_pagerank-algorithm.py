# -*- coding: utf-8 -*-
"""
#***********************************************************
# PAGERANK ALGORITHM 
#***********************************************************

Created on Mon Mar 26 11:53:55 2018
@author: nahmad3
"""


# Before we begin, let's load the libraries.
%pylab notebook
import numpy as np
import numpy.linalg as la
from readonly.PageRankFunctions import *  # not working? 
np.set_printoptions(suppress=True)



# example of link matrix for 6 websites, A to F: 
L = np.array([[0,   1/2, 1/3, 0, 0,   0 ],
              [1/3, 0,   0,   0, 1/2, 0 ],
              [1/3, 1/2, 0,   1, 0,   1/2 ],
              [1/3, 0,   1/3, 0, 1/2, 1/2 ],
              [0,   0,   0,   0, 0,   0 ],
              [0,   0,   1/3, 0, 0,   0 ]])


'''
In principle, we could use a linear algebra library, as below, to calculate 
the eigenvalues and vectors. And this would work for a small system. 
But this gets unmanagable for large systems. 
And since we only care about the principal eigenvector (the one with the 
largest eigenvalue, which will be 1 in this case), we can use the power 
iteration method which will scale better, and is faster for large systems.

Use the code below to peek at the PageRank for this micro-internet.
'''

# calculating pagerank by calculating all eValues and eVectors: 
eVals, eVecs = la.eig(L) # Gets the eigenvalues and vectors
order = np.absolute(eVals).argsort()[::-1] # Orders them by their eigenvalues
eVals = eVals[order]
eVecs = eVecs[:,order]

r = eVecs[:, 0] # Sets r to be the principal eigenvector
100 * np.real(r / np.sum(r)) # Make this eigenvector sum to one, then multiply
                                # by 100 Procrastinating Pats

'''
Let's now try to get the same result using the Power-Iteration method that was 
covered in the video. This method will be much better at dealing with large
 systems.

First let's set up our initial vector,  r(0)r(0) , so that we have our 100 
Procrastinating Pats equally distributed on each of our 6 websites.
'''

r = 100 * np.ones(6) / 6 # Sets up this vector (6 entries of 1/6 × 100 each)
r # Shows it's value


# iterate 100 times using for loop: 
r = 100 * np.ones(6) / 6 # Sets up this vector (6 entries of 1/6 × 100 each)
for i in np.arange(100) : # Repeat 100 times
    r = L @ r
r

# Or even better, we can keep running until we get to the required tolerance.
r = 100 * np.ones(6) / 6 # Sets up this vector (6 entries of 1/6 × 100 each)
lastR = r
r = L @ r
i = 0
while la.norm(lastR - r) > 0.01 :
    lastR = r
    r = L @ r
    i += 1
print(str(i) + " iterations to convergence.")
r























