# -*- coding: utf-8 -*-
#*********************************************
# PYTHON EXPERIMENTS 
#*********************************************

import numpy as np
import matplotlib.pyplot as plt




#**************************************
# Basic arithmetic 
#**************************************
2**3 # exponent
2//3 # floor division 
3//2

2/3  # division

2%3  # modulo 
10%3

x = 4 + 5 
y=x-2
y




#***************************************
# Primitive data types 

# references:
# https://lectures.quantecon.org/py/python_essentials.html 
#***************************************

# boolean: 
x = True
type(x)


# integers and floats: 
a = 2
b = 2.1
type(a)
type(b)









#**************************************
# Containers: Lists 
#**************************************

values = []  # create empty list

x = [10, 20, 'hi', "yes"]  # heterogeneous list 
type(x)
print(x)

# note that lists are zero-indexed: 
x[0] == 10 
x[2] == "hi"

x[1:]  # subset the list from index 1 to end (left endpoint INCLUDED)
x[:2]  # subset the list from index 0 to index 2 (right endpoint EXCLUDED)

x[1:3]


# list methods: ----------
# append() is a 'method', which is a function "attached to" an object - in this
# case, the list x. 
x.append("yo")
print(x)

# pop() method: 
# Remove the item at the given position in the list, and 
# return it. If no index is specified, a.pop() removes and 
# returns the last item in the list.
x.pop(); x  # you don't need to say "print(x)" 
x.pop(0); x
x.pop(1); x

# more list methods: --------
# here's the full list: https://docs.python.org/3/tutorial/datastructures.html#more-on-lists 

x.pop(); x
x.append('new'); x
x.append('hi'); x
x.insert(0, "first"); x
x.remove("first"); x
x.index("hi")
x.count(10)
x.sort(); x  # only works for all numeric elements 


# example: 
fruits = ['orange', 'apple', 'pear', 'banana', 'kiwi', 'banana']
fruits.index('banana')  # first instance of banana is at position 3 (counting from 0)
fruits.index('banana', 4)  # second banana is at position 5; argument "4" used to say, start searching the list from that position



# fill an empty list with a for loop: 
list1 = []

for i in range(12): 
    print("Number being appended is: " + str(i))
    list1.append(i)  # note that we don't need an explicit assignment statement such as list1 = list1.append(i)

list1




# List comprehensions -----
# List comprehensions are an elegant Python tool for creating lists.

# Common applications are to make new lists where each element
# is the result of some operations applied to each member of another sequence 
# or iterable

# reference: https://docs.python.org/3/tutorial/datastructures.html

'''
A list comprehension consists of an expression followed 
by a for clause, then zero or more for or if clauses - all of which are enclosed 
in square brackets. 

The result will be a new list resulting from evaluating the expression in the 
context of the for and if clauses which follow it.

Example: instead of the indented "for loop" structure above, use: 
'''

list2 = [i for i in range(12)]
list2

list1 == list2


# partial fizzbuzz: 
ans = [str(i) + " fizzbuzz" for i in range(101) if i%15 == 0]
ans

# full fizzbuzz: 
for i in range(101): 
    if i%15 == 0: 
        print("fizzbuzz")
    elif i%3 == 0 : 
        print("fizz")
    elif i%5 == 0: 
        print("buzz") 
    else: 
        print(i) 


        
    
    



#**************************************
# Strings 
#**************************************














#**************************************
# Functions 
#**************************************


def addtwo (num): 
    # takes a num, adds 2 if it's not 2 
    if num != 2: 
        return(num + 2)
    else: 
        return('nope')
        
addtwo(4)
addtwo(2)



# lambda functions: -----------------
# these are unnamed functions that are defined "inline": 
squares = list(map(lambda x: x**2, range(10)))
squares

# the x after lambda specifies name of the argument to the function, the 
# operations after the colon specify what to do with the argument. 

# Note: the above can be better achieved as follows: 
squares2 = [x**2 for x in range(10)]

squares == squares2  # True 



#****************************************
# Example: plot a white noise process
#**************************************** 

x = np.random.randn(100); x  # random is a subpackage within the numpy package

plt.plot(x)
plt.show()






#****************************************  
# Reading in and outputting files 

# references:
# https://lectures.quantecon.org/py/python_essentials.html 
#****************************************  
# where is the working directory?
%pwd  # "print working directory?" 

# read a text file: 
data = open("D:/python-notes/data/test.txt", "r")

"""When you use the open function, it returns something called 
a file object. File objects contain methods and attributes 
that can be used to collect information about the file 
you opened. They can also be used to manipulate said file.
"""

# display the contents of the text file as a string:  
print(data.read())


