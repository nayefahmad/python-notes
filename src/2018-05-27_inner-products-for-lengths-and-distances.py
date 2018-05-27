# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:31:06 2018

@author: Nayef
"""

#*********************************************************************
# USING INNER PRODUCTS FOR LENGTHS/DISTANCES OF VECTORS 
#*********************************************************************

import numpy as np



# Q1 ---------------------------------------------------------------------
x = np.array([1, -1, 3])

M = np.array([[2, 1, 0],
              [1, 2, -1], 
              [0, -1, 2]])
    

# define inner prod fn: 
def innerprod(x, y, matrix): 
    return np.transpose(x) @ matrix @ y

# call the fn: 
innerprod(x, x, M)



# Q2 ---------------------------------------------------------------------

x = np.array([.5, -1, -.5])
y = np.array([0, 1, 0])

M = np.array([[2, 1, 0],
              [1, 2, -1], 
              [0, -1, 2]])

innerprod(x-y, x-y, M)


# Q3 ---------------------------------------------------------------------
x = np.array([-1, 1])

M = np.array([[5, -1], 
              [-1, 5]])

M = .5 * M

innerprod(x, x, M)


# Q4 ---------------------------------------------------------------------
x = np.array([.5, -1, -.5])
y = np.array([0, 1, 0])

M = np.array([[2, 1, 0],
              [1, 2, -1], 
              [0, -1, 2]])

innerprod(x-y, x-y, M)

dist = np.sqrt(innerprod(x-y, x-y, M))
dist


# Q5 ---------------------------------------------------------------------
x = np.array([-1, -1, -1])

np.transpose(x) @ x



