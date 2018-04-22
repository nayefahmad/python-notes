# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 13:53:09 2018

@author: Nayef
"""

#***************************************************
# SIMPLE NEURAL NETWORKS
#***************************************************

import numpy as np
from numpy.linalg import norm, inv
from numpy import transpose


#****************************************************************
# 2 node neural network that acts as a NOT gate: ----------------
#****************************************************************

# First we set the state of the network
σ = np.tanh  # this is the sigmoid activation fn we're using 
w1 = -5  
b1 = 5

# Then we define the neuron activation.
def a1(a0) :
  return σ(w1 * a0 + b1)
  
# Finally let's try the network out!
# Replace x with 0 or 1 below,
a1(1)





#****************************************************************
# More complicated network: ----------------
#****************************************************************
# First set up the network.
sigma = np.tanh
W = np.array([[-2, 4, -1],[6, 0, -3]])
b = np.array([0.1, -2.5])

# Define our input vector
x = np.array([0.3, 0.4, 0.1])

# define fn to get scalar outputs: 
def calc_output(input_vec, 
                weight_matrix, 
                bias_vector, 
                index): 
    
    # index starts at 0 to represent first output
    
    return(sigma((W[index] @ input_vec) + bias_vector[index]))

# evaluate first output, a1_0: 
a1_0 = calc_output(x, W, b, 0)

# evaluate second output, a1_1
a1_1 = calc_output(x, W, b, 1)    


    
a1 = np.array([a1_0, a1_1])
print(a1)




#**********************************************************
# Derivatives of the cost fn of the network: 
#**********************************************************





















