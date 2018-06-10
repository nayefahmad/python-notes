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



def projection_matrix_1d(b):
    """Compute the projection matrix onto the space spanned by `b`
    Args:
        b: ndarray of dimension (D,), the basis for the subspace
    
    Returns:
        P: the projection matrix
    """
    P = np.transpose(b) @ b
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

# Projection onto general subspace
# ===YOU SHOULD EDIT THIS FUNCTION===
def projection_matrix_general(B):
    """Compute the projection matrix onto the space spanned by `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace
    
    Returns:
        P: the projection matrix
    """
    P = np.eye(B.shape[0]) # EDIT THIS
    return P

# ===YOU SHOULD EDIT THIS FUNCTION===
def project_general(x, B):
    """Compute the projection matrix onto the space spanned by `B`
    Args:
        B: ndarray of dimension (D, E), the basis for the subspace
    
    Returns:
        y: projection of x in space spanned by b
    """
    p = np.zeros(x.shape) # EDIT THIS
    return p