# -*- coding: utf-8 -*-
"""
Implementing backtracking 
Created on Sat Nov 30 18:32:41 2019

@author: Nayef
"""

import numpy as np 
from functools import partial

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
            if safe_up_to(solution):
                if position >= size-1:
                    yield np.array(solution)
                else: 
                    yield from extend_solution(position+1)
        solution[position] = None

    return extend_solution(0)




#%% Function to check partial solutions 
    
def safe_up_to(target, partial_solution): 
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
    
    
    
#%% Call the backtracking function

# correct way to generate outputs: 
for sol in solve(values=range(10), safe_up_to=partial(safe_up_to, 4), size=7):
    print(sol, sol.sum())


# what if I just call solve()?
# solve(values=range(10), safe_up_to=partial(safe_up_to, 4), size=7)
#Out[38]: <generator object solve.<locals>.extend_solution at 0x00000176AC39EF10>

# Can we use a listcomp? 
# [sol in solve(values=range(10), safe_up_to=partial(safe_up_to, 4), size=7)]
# ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

# can we create a list of lists?
#solutions = []
#i = 0
#for sol in solve(values=range(10), safe_up_to=partial(safe_up_to, 4), size=7):
#    solutions[i] = sol
#    i += i


