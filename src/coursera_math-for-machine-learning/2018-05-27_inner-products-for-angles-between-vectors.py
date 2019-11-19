# -*- coding: utf-8 -*-
"""
Created on Sun May 27 16:34:56 2018

@author: Nayef
"""

#***************************************************************
# USING INNER PRODUCTS TO FIND ANGLES BETWEEN VECTORS 
#***************************************************************

import numpy as np

'''
# NOTE: 
Since we're not using standard dot product to find angles, 
results will be a little weird. 
Ex: normally we think of angle between [1,1] and [-1, 1] as 
90 degrees, but this isn't necessarily true when using other r
definitions of the inner product. 

'''

# def fn to find angles:
def cos_angle(x, y, M):
    
    # define inner prod fn: 
    def innerprod(x, y, matrix): 
        return np.transpose(x) @ matrix @ y

    return innerprod(x, y, M)/(np.sqrt(innerprod(x, x, M)) * np.sqrt(innerprod(y, y, M)))


# Q1 ------------------------------------------------------------------------
x = np.array([1,1])
y = np.array([-1, 1])

M = np.array([[2, -1], 
              [-1, 4]])


a = cos_angle(x, y, M)
angle = np.arccos(a)
angle

# Q2 ------------------------------------------------------------------------
x = np.array([0, -1])
y = np.array([1, 1])

M = np.array([[1, -1/2], 
              [-1/2, 5]])


a = cos_angle(x, y, M)
angle = np.arccos(a)
angle


# Q3 ------------------------------------------------------------------------
x = np.array([2, 2])
y = np.array([-2, -2])

M = np.array([[2, 1], 
              [1, 4]])


a = cos_angle(x, y, M)
angle = np.arccos(a)
angle


# Q4 ------------------------------------------------------------------------
x = np.array([1, 1])
y = np.array([1, -1])

M = np.array([[1, 0], 
              [0, 5]])


a = cos_angle(x, y, M)
angle = np.arccos(a)
angle



# Q5 ------------------------------------------------------------------------
x = np.array([1, 1, 1])
y = np.array([2, -1, 0])

M = np.array([[1, 0, 0], 
              [0, 2, -1], 
              [0, -1, 3]])


a = cos_angle(x, y, M)
angle = np.arccos(a)
angle












