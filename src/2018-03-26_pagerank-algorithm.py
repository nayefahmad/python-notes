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





#*******************************************************************
# WHAT IF WE ADD AN ABSORBING STATE IN THE MARKOV CHAIN? (use damping) 
#*******************************************************************

# Add a new site G that you get to from F, but which you can't leave. 
# New link matrix: 
    
 # We'll call this one L2, to distinguish it from the previous L.
L2 = np.array([[0,   1/2, 1/3, 0, 0,   0, 0 ],
               [1/3, 0,   0,   0, 1/2, 0, 0 ],
               [1/3, 1/2, 0,   1, 0,   1/3, 0 ],
               [1/3, 0,   1/3, 0, 1/2, 1/3, 0 ],
               [0,   0,   0,   0, 0,   0, 0 ],
               [0,   0,   1/3, 0, 0,   0, 0 ],
               [0,   0,   0,   0, 0,   1/3, 1 ]])

r = 100 * np.ones(7) / 7 # Sets up this vector (6 entries of 1/6 × 100 each)
lastR = r
r = L2 @ r
i = 0
while la.norm(lastR - r) > 0.01 :
    lastR = r
    r = L2 @ r
    i += 1
print(str(i) + " iterations to convergence.")
r    


'''
That's no good! G seems to be taking all the traffic on the micro-internet,
 and somehow coming at the top of the PageRank. This behaviour can be 
 understood, because once a Pat get's to Geoff's Website, they can't leave, 
 as all links head back to Geoff.

To combat this, we can add a small probability that the Procrastinating Pats
 don't follow any link on a webpage, but instead visit a website on the 
 micro-internet at random. We'll say the probability of them following a link 
 is  d  and the probability of choosing a random website is therefore  1−d.
 We can use a new matrix to work out where the Pat's visit each minute.
 '''

d = 0.5 # Feel free to play with this parameter after running the code once.
M = d * L2 + (1-d)/7 * np.ones([7, 7]) # np.ones() is the J matrix, with ones for each entry.
    

r = 100 * np.ones(7) / 7 # Sets up this vector (6 entries of 1/6 × 100 each)
lastR = r
r = M @ r
i = 0
while la.norm(lastR - r) > 0.01 :
    lastR = r
    r = M @ r
    i += 1
print(str(i) + " iterations to convergence.")
r



'''
This is certainly better, the PageRank gives sensible numbers for the 
Procrastinating Pats that end up on each webpage. This method still predicts 
Geoff has a high ranking webpage however. This could be seen as a consequence 
of using a small network. We could also get around the problem by not countin
self-links when producing the L matrix (an if a website has no outgoing links, 
make it link to all websites equally). We won't look further down this route, 
as this is in the realm of improvements to PageRank, rather than eigenproblems.

You are now in a good position, having gained an understanding of PageRank, 
to produce your own code to calculate the PageRank of a website with thousands 
of entries.
'''















