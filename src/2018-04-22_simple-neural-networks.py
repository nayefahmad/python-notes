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









