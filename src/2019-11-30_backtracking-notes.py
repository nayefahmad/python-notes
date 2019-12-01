# -*- coding: utf-8 -*-
"""
Implementing backtracking 
Created on Sat Nov 30 18:32:41 2019

@author: Nayef
"""

import numpy as np 

#%% Backtracking "engine" function 

def solve(values, safe_up_to, size):
    """Finds a solution to a backtracking problem.

    values     -- a sequence of values to try, in order. For a map coloring
                  problem, this may be a list of colors, such as ['red',
                  'green', 'yellow', 'purple']
    safe_up_to -- a function with two arguments, solution and position, that
                  returns whether the values assigned to slots 0..pos in
                  the solution list, satisfy the problem constraints.
    size       -- the total number of “slots” you are trying to fill

    Return the solution as a list of values.
    """
    solution = [None] * size

    def extend_solution(position):
        for value in values:
            solution[position] = value
            if safe_up_to(solution, position):
                if position >= size-1 or extend_solution(position+1):
                    return solution
        return None

    return extend_solution(0)




#%% Function to check partial solutions 
    
def safe_up_to(partial_solution, target = 100): 
    """
    Checks that a partial solution (string of numerals) sums to less than 10
    
    Partial soln is passed to the function as a list: e.g. [1, 5]
    
    """
    partial_solution = np.array(partial_solution)  # convert to np array 
    
    # replace None with NaN
    partial_solution = np.where(partial_solution == None, np.nan, partial_solution)
    
    if np.nansum(partial_solution) <= target: 
        return True
    else: 
        return False 
    
    
    
#%% 


