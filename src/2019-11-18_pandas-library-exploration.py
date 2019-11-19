# -*- coding: utf-8 -*-
"""
Pandas exploration 
Created on Mon Nov 18 16:48:26 2019

@author: nahmad3
"""

#%% lib
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm

# %matplotlib 

#%% Create a pd dataframe: examples from docs 

d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)

df.dtypes



#%% 

# Create a pd dataframe 

col1 = np.arange(1, 10)
col2 = np.random.rand(10)

np.c_[[col1], [col2]]

df1 = pd.DataFrame(data = np.c_[[col1], [col2]])

df1.shape

