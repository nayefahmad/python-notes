# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 19:54:57 2018

@author: Nayef
"""

#****************************************************
# PROJECTIONS USING MATRICES 
#****************************************************

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import numpy as np
import numpy.testing as np_test



#****************************************************
# PROJECTION ONTO 1D SUBSPACES 
#****************************************************

def projection_matrix_1d(b):
    """Compute the projection matrix onto the space spanned by `b`
    Args:
        b: ndarray of dimension (D,), the basis for the subspace
    
    Returns:
        P: the projection matrix
    """
    P_numerator = np.outer(b, np.transpose(b)) 
    P_denom = np.linalg.norm(b)**2 
    P = np.divide(P_numerator, P_denom)
    
    return P

# test the function: 
projection_matrix_1d(np.array([1, 2, 2]))


def project_1d(x, b):
    """Compute the projection matrix onto the space spanned by `b`
    Args:
        x: the vector to be projected
        b: ndarray of dimension (D,), the basis for the subspace
    
    Returns:
        y: projection of x in space spanned by b
    """
    p = projection_matrix_1d(b) @ x 
    return p

    

    
#****************************************************
# PROJECTION ONTO GENERAL N-DIMENSIONAL SUBSPACES 
#****************************************************

def projection_matrix_general(B):
    """Compute the projection matrix onto the space spanned by `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace
    
    Returns:
        P: the projection matrix
    """
    P = B @ np.linalg.inv(np.transpose(B) @ B) @ np.transpose(B) 
    return P

def project_general(x, B):
    """Compute the projection matrix onto the space spanned by `B`
    Args:
        B: ndarray of dimension (D, E), the basis for the subspace
    
    Returns:
        y: projection of x in space spanned by b
    """
    p = projection_matrix_general(B) @ x 
    return p
    
    
    
    
#****************************************************
# TESTING THE PROJECTIONS: 
#****************************************************

# Orthogonal projection in 2d
# define basis vector for subspace
b = np.array([2,1]).reshape(-1,1); print(b)  # todo: what does reshape do here? 
# point to be projected later
x = np.array([1,2]).reshape(-1, 1); print(x) 


# Test 1D
x = projection_matrix_1d(np.array([1, 2, 2]))
print(x)

np_test.assert_almost_equal(projection_matrix_1d(np.array([1, 2, 2])), 
                            np.array([[1,  2,  2],
                                      [2,  4,  4],
                                      [2,  4,  4]]) / 9)

np_test.assert_almost_equal(project_1d(np.ones(3),
                                       np.array([1, 2, 2])),
                            np.array([5, 10, 10]) / 9)

B = np.array([[1, 0],
              [1, 1],
              [1, 2]])

# Test General
np_test.assert_almost_equal(projection_matrix_general(B), 
                            np.array([[5,  2, -1],
                                      [2,  2,  2],
                                      [-1, 2,  5]]) / 6)

np_test.assert_almost_equal(project_general(np.array([6, 0, 0]), B), 
                            np.array([5, 2, -1]))
print('correct')












