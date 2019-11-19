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

%matplotlib inline  # other option is: `%matplotlib qt5`



#%% Create a pd dataframe: examples from docs 

d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)

df.dtypes



#%% Create a pd dataframe from np arrays 

np.random.seed(10)
col1 = np.arange(0, 10)
col2 = np.random.rand(10)

type(col1)
type(col2)
len(col1)
#dir(col1)


# Option 1: 
# call (pd.DataFrame) with a dictionary passed to the `data` arg 
df1 = pd.DataFrame(data = {'col_1':col1, 'col_2':col2})

# Option 2: 
# 
# np.concatenate((col1, col2), axis=1)  # doesn't work 
np.stack((col1, col2), axis = 1)  # works, but forces all numbers to be floats? 

df2 = pd.DataFrame(data = np.stack((col1, col2), axis = 1), 
                   columns = ['col1', 'col2'])

df1
df2







#%% Examinig the pd dataframe 
df1.shape
df1.describe
df1.info
df1.dtypes
df1.columns  # colnames 
df1.columns[1]
df1[:-5]  # drop last 5 rows 


df2.dtypes  # note that both cols were coerced to floats when we stacked the np arrays 


#%%Plots 

df1.plot.line()  # note that we need the parens at the end here 

plt.plot(['col_1'], ['col_2'], 'ro--')
