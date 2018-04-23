# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:47:24 2018

@author: Nayef
"""

#********************************************************************
# Backpropagation example  
#********************************************************************
'''
In this assignment, you will train a neural network to draw a curve. 
The curve takes one input variable, the amount travelled along the curve 
from 0 to 1, and returns 2 outputs, the 2D coordinates of the position of 
points on the curve.

To help capture the complexity of the curve, we shall use two hidden layers 
in our network with 6 and 7 neurons respectively.
'''


'''
Recall from assignments in the previous course in this specialisation that 
matrices can be multiplied together in two ways.

Element wise: when two matrices have the same dimensions, matrix elements in 
the same position in each matrix are multiplied together In python this uses 
the ' ∗∗ ' operator.

A = B * C
Matrix multiplication: when the number of columns in the first matrix is the 
same as the number of rows in the second. In python this uses the ' @ '
 operator

A = B @ C

''' 





# PACKAGE
# First load the worksheet dependencies.
import numpy as np
import matplotlib.pyplot as plt

# Here is the activation function and its derivative.
sigma = lambda z : 1 / (1 + np.exp(-z))
d_sigma = lambda z : np.cosh(z/2)**(-2) / 4

# This function initialises the network with it's structure, it also resets 
# any training already done.
def reset_network (n1 = 6, n2 = 7, random=np.random) :
    global W1, W2, W3, b1, b2, b3
    
    # weights: 
    W1 = random.randn(n1, 1) / 2
    W2 = random.randn(n2, n1) / 2
    W3 = random.randn(2, n2) / 2
    
    # biases: 
    b1 = random.randn(n1, 1) / 2
    b2 = random.randn(n2, 1) / 2
    b3 = random.randn(2, 1) / 2

# This function feeds forward each activation to the next layer. It 
# returns all weighted sums and activations.
def network_function(a0) :
    z1 = W1 @ a0 + b1
    a1 = sigma(z1)
    z2 = W2 @ a1 + b2
    a2 = sigma(z2)
    z3 = W3 @ a2 + b3
    a3 = sigma(z3)
    return a0, z1, a1, z2, a2, z3, a3

# This is the cost function of a neural network with respect to a training set.
def cost(x, y) :
    return np.linalg.norm(network_function(x)[-1] - y)**2 / x.size


                          
                          
#********************************************************************
# Jacobian of the Cost fn with respect to the third layer weights and biases. 
#********************************************************************

def J_W3 (x, y) :
    # First get all the activations and weighted sums at each layer of the network.
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    # We'll use the variable J to store parts of our result as we go along, 
    # updating it in each line.
    
    # Firstly, we calculate dC/da3, using the expressions above.
    J = 2 * (a3 - y)
    
    # Next multiply the result we've calculated by the derivative of sigma, 
    # evaluated at z3.
    J = J * d_sigma(z3)
    
    # Then we take the dot product (along the axis that holds the training 
    # examples) with the final partial derivative,
    # i.e. dz3/dW3 = a2
    # and divide by the number of training examples, for the average over all 
    # training examples.
    J = J @ a2.T / x.size  # .T is for transpose? 
    # Finally return the result out of the function.
    return J

    
# In this function, you will implement the jacobian for the bias.
# As you will see from the partial derivatives, only the last partial derivative is different.
# The first two partial derivatives are the same as previously.
# ===YOU SHOULD EDIT THIS FUNCTION===
def J_b3 (x, y) :
    # As last time, we'll first set up the activations.
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    # Next you should implement the first two partial derivatives of the Jacobian.
    # ===COPY TWO LINES FROM THE PREVIOUS FUNCTION TO SET UP THE FIRST TWO JACOBIAN TERMS===
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    
    # For the final line, we don't need to multiply by dz3/db3, because that 
    # is multiplying by 1.
    # We still need to sum over all training examples however.
    # There is no need to edit this line.
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J
    




#********************************************************************
# Jacobian of the Cost fn with respect to the SECOND layer weights and biases. 
#********************************************************************    
# Compare this function to J_W3 to see how it changes.
# There is no need to edit this function.
def J_W2 (x, y) :
    #The first two lines are identical to in J_W3.
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)    
    J = 2 * (a3 - y)
    
    # the next two lines implement da3/da2, first σ' and then W3.
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    
    # then the final lines are the same as in J_W3 but with the layer number bumped down.
    J = J * d_sigma(z2)  # z2 instead of z3 
    J = J @ a1.T / x.size  # a1 instead of a2 
    return J

# As previously, fill in all the incomplete lines.
# ===YOU SHOULD EDIT THIS FUNCTION===
def J_b2 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    
    # first find dC/da3
    J = 2 * (a3 - y)
    
    # the next two lines implement da3/da2, first σ' and then W3.
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    
    # then find da2/dz2
    J = J * d_sigma(z3)
    
    # dz/db = 1, so no need to multiply that 
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J

    
    
    
    
    
    
    
    
























