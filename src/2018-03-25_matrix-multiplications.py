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




