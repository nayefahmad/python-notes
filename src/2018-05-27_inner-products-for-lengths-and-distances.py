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


# define inner prod fn: 
def innerprod(x): 
    M = np.array([[2, 1, 0],
                  [1, 2, 1], 
                  [0, -1, 2]])
    
    return np.transpose(x) @ M @ x 

# call the fn: 
innerprod(x)
