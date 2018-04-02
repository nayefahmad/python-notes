# -*- coding: utf-8 -*-
"""
# *********************************************************
# MATRIX MULTIPLIACTIONS 
# *********************************************************

Created on Sun Mar 25 20:28:06 2018
@author: Nayef
"""

import numpy as np
from numpy.linalg import norm, inv
from numpy import transpose


# Q2 *****************************************
T = np.array([[2, 7],
              [0, -1]])

C = np.array([[7, 1],
              [-3, 0]])

Cinv = inv(C)
# print(Cinv)

D = Cinv @ T @ C
# print(D)  # looks like we're getting float point errors 


# template: *****************************************
Dcube = np.array([[125, 0],
                  [0, 64]])

C = np.array([[1, 1],
              [1, 2]])

Cinv = inv(C)
print(Cinv)

T = C @ Dcube @ Cinv
print(T) 



#*****************************************************
# random calculations: -------------------------------
#*****************************************************

# WEEK 5 QUIZ ***************************************
# QUESTION 1: 

# Eigenvalues
M = np.array([[1, 0, 0],
              [0, 2, 0],
              [0, 0, 3]])
vals, vecs = np.linalg.eig(M)
vals
vecs

# the eigenvectors of M are the original basis vectors: they are simply 
#   scaled by factors 1, 2, 3; no change in direction. 

M = np.array([[4, -5, 6],
              [7, -8, 6],
              [3/2, -1/2, -2]])

M @ np.array([-3, -3, -1]) == np.array([-3, -3, -1])  # is eigenvec
M @ np.array([.5, -.5, -1]) == np.array([.5, -.5, -1])  # not eigenvec
M @ np.array([-3,-2,1]) == np.array([-3,-2,1])  # not eigenvec
M @ np.array([-1,1,-2]) == np.array([-1,1,-2]) # not eigenvec
M @ np.array([-2/sqrt(9), -2/sqrt(9), 1/sqrt(9)])  == np.array([-2/sqrt(9), -2/sqrt(9), 1/sqrt(9)])  # not eigenvec
M @ np.array([1/sqrt(6), -1/sqrt(6), 2/sqrt(6)]) ==  np.array([1/sqrt(6), -1/sqrt(6), 2/sqrt(6)])  # not eigenvec



# QUESTION 2: 
L = np.array([[0, 0, 0, 1 ],
              [1, 0, 0, 0 ],
              [0, 1, 0, 0 ],
              [0, 0, 1, 0 ]])
    
L_vals, L_vecs = np.linalg.eig(L)
L_vals  # some eigenvalues are complex 
L_vecs  # this is a 4 x 4 matrix, but with complex entries. 



# QUESTION 3: --------------------------
L_prime = np.array([[0.1, 0.1, 0.1, 0.7],
                    [0.7, 0.1, 0.1, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.1, 0.1, 0.7, 0.1]])
        
L_prime_vals, L_prime_vecs = np.linalg.eig(L_prime)
L_prime_vals  # some eigenvalues are complex 
L_prime_vecs  # this is a 4 x 4 matrix, but with complex entries. 



# QUESTION 4: BLOCK DIAGONAL LINK MATRIX: ---------------------
L_block = np.array([[0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]])
    
L_block_vals, L_block_vecs = np.linalg.eig(L_block)
L_block_vals  # some eigenvalues are complex; none is equal to 1.0 
L_block_vecs  # this is a 4 x 4 matrix, but with complex entries. 

np.linalg.det(L_block)  # non-zero determinant 


# let's add damping:
L_block_damped = np.array([[0.1, 0.7, 0.1, 0.1],
                           [0.7, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.7],
                           [0.1, 0.1, 0.7, 0.1]])

L_block_damp_vals, L_block_damp_vecs = np.linalg.eig(L_block_damped)
L_block_damp_vals  
L_block_damp_vecs 



# QUESTION 8: FIND EIGENVECTORS: -------------------------------

M = np.array([[3/2, -1],
              [-.5, .5]])
M_vals, M_vecs = np.linalg.eig(M)
M_vals
M_vecs

v1 = np.array([1-sqrt(5), 1])
v2 = np.array([-1-sqrt(3), 1])
v3 = np.array([-1-sqrt(5), 1])
v4 = np.array([1-sqrt(3), 1])

# check v1
M @ v1 == (v1 * M_vals[0])
M @ v1 == (v1 * M_vals[1])

# check v2: 
M @ v2 == (v2 * M_vals[0])
M @ v2 == (v2 * M_vals[1])

# check v3
M @ v3 == (v3 * M_vals[0])
M @ v3 == (v3 * M_vals[1])

# check v4: 
M @ v4 == (v4 * M_vals[0])
M @ v4 == (v4 * M_vals[1])



# QUESTION 10: FIND A-squared: ------------------




    
    


