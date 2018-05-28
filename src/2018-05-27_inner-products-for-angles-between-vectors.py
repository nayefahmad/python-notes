# -*- coding: utf-8 -*-
"""
Created on Sun May 27 16:34:56 2018

@author: Nayef
"""

#***************************************************************
# USING INNER PRODUCTS TO FIND ANGLES BETWEEN VECTORS 
#***************************************************************

import numpy as np


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

















