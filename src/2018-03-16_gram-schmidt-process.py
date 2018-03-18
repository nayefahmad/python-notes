# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 13:06:24 2018

@author: nahmad3
"""

# GRADED FUNCTION
import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14 # That's 1×10⁻¹⁴ = 0.00000000000001

# Our first function will perform the Gram-Schmidt procedure for 4 basis vectors.
# We'll take this list of vectors as the columns of a matrix, A.
# We'll then go through the vectors one at a time and set them to be orthogonal
#   to all the vectors that came before it. Before normalising.
# Follow the instructions inside the function at each comment.
# You will be told where to add code to complete the function.

# function definition: ---------
def gsBasis4(A) :
    B = np.array(A, dtype=np.float_) # Make B as a copy of A, since we're going to alter it's values.
    # The zeroth column is easy, since it has no other vectors to make it normal to.
    # All that needs to be done is to normalise it. I.e. divide by its modulus, or norm.
    
    print("original B[:,0]=", B[:,0])
    # print(la.norm(B[:,0]))
    B[:, 0] = B[:, 0] / la.norm(B[:, 0])
    print("final B[:,0]=",B[:,0])


    # For the first column, we need to subtract any overlap with our new zeroth vector.
    print("original B[:,1]=", B[:,1])
    print("dot prod: B[:, 1] @ B[:, 0]=", B[:, 1] @ B[:, 0])
    print("^^this doesn't seem right")  # todo: 
    print("vector result: (B[:, 1] @ B[:, 0]) * B[:, 0]=", B[:, 1] @ B[:, 0] * B[:, 0])
    
    B[:, 1] = B[:, 1] - (B[:, 1] @ B[:, 0]) * B[:, 0]
    print("final B[:, 1]=", B[:, 1])
    # @ is an infix operator for matrix multiplication (dot product in this case)
    # see https://www.python.org/dev/peps/pep-0465/ 
    
    # * does element-wise multiplication of a scalar with each element of a vector
    
    
    # If there's anything left after that subtraction, then B[:, 1] is linearly independant of B[:, 0]
    # If this is the case, we can normalise it. Otherwise we'll set that vector to zero.
    if la.norm(B[:, 1]) > verySmallNumber :  # i.e. approx not zero
        B[:, 1] = B[:, 1] / la.norm(B[:, 1])
    else :
        B[:, 1] = np.zeros_like(B[:, 1])
    
    
    # Now we need to repeat the process for column 2.
    # Insert two lines of code, the first to subtract the overlap with the zeroth vector,
    # and the second to subtract the overlap with the first.
    
    
    # Again we'll need to normalise our new vector.
    # Copy and adapt the normalisation fragment from above to column 2.
    
    
    
    
    # Finally, column three:
    # Insert code to subtract the overlap with the first three vectors.
    
    
    
    # Now normalise if possible
    
    
    
    
    # Finally, we return the result:
    return B



#***********************************************************************************
#  PART TWO 
#***********************************************************************************

# The second part of this exercise will generalise the procedure.
# Previously, we could only have four vectors, and there was a lot of repeating in the code.
# We'll use a for-loop here to iterate the process for each vector.
def gsBasis(A) :
    B = np.array(A, dtype=np.float_) # Make B as a copy of A, since we're going to alter it's values.
    # Loop over all vectors, starting with zero, label them with i
    for i in range(B.shape[1]) :
        # Inside that loop, loop over all previous vectors, j, to subtract.
        for j in range(i) :
            # Complete the code to subtract the overlap with previous vectors.
            # you'll need the current vector B[:, i] and a previous vector B[:, j]
            B[:, i] = 
        # Next insert code to do the normalisation test for B[:, i]
        if :
            
        
            
    # Finally, we return the result:
    return B

# This function uses the Gram-schmidt process to calculate the dimension
# spanned by a list of vectors.
# Since each vector is normalised to one, or is zero,
# the sum of all the norms will be the dimension.
def dimensions(A) :
    return np.sum(la.norm(gsBasis(A), axis=0))




#***********************************************************************************
# TEST THE FUNCTION
#***********************************************************************************

V = np.array([[1,0,2,6],
              [0,1,8,2],
              [2,8,3,1],
              [1,-6,2,3]], dtype=np.float_)
gsBasis4(V)



# Once you've done Gram-Schmidt once,
# doing it again should give you the same result. Test this:
U = gsBasis4(V)
gsBasis4(U)


# Try the general function too.
gsBasis(V)


# See what happens for non-square matrices
A = np.array([[3,2,3],
              [2,5,-1],
              [2,4,8],
              [12,2,1]], dtype=np.float_)
gsBasis(A)

dimensions(A)

