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

%matplotlib qt5
#%matplotlib inline  # other option is: `%matplotlib qt5`s
plt.rcParams['figure.figsize'] = [8, 8]  # default fig size in jupyter notebooks; doesn't work here? 


#%% Create a pd dataframe: examples from docs 

d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)

df.dtypes



#%% Create a pd dataframe from np arrays 

np.random.seed(10)
col1 = np.arange(0, 10)
col2 = np.random.rand(10)
col3 = np.random.rand(10) *2

type(col1)
type(col2)
type(col3)
len(col1)
#dir(col1)


# Option 1: 
# call (pd.DataFrame) with a dictionary passed to the `data` arg 
df1 = pd.DataFrame(data = {'col_1':col1, \
                           'col_2':col2, \
                           'col_3':col3})

# Option 2: 
# 
# np.concatenate((col1, col2), axis=1)  # doesn't work 
np.stack((col1, col2, col3), axis = 1)  
# works, but forces all numbers to be floats? 
# axis = 0 is rows; axis = 1 is cols  

df2 = pd.DataFrame(data = np.stack((col1, col2, col3), axis = 1), 
                   columns = ['col1', 'col2', 'col3'])

df1
df2

df1.dtypes
df2.dtypes  # note that both cols were coerced to floats when we stacked the np arrays 






#%% Examinig the pd dataframe 

# Every pd df has 3 key components: index, columns, values  
# reference: https://medium.com/dunder-data/selecting-subsets-of-data-in-pandas-6fcd0170be9c

df1.index  # axis = 0; RangeIndex is the default when we dont set one of the columns as the index 
df1.columns  # axis = 1; colnames 
df1.values  # data in the df 

# change the index:
df1.index = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10']
df1
df1.index  # obj of type 'Index' 

df1.shape
df1.describe
df1.info
df1.dtypes





#%% subsetting the df 
# reference: https://medium.com/dunder-data/selecting-subsets-of-data-in-pandas-6fcd0170be9c

# Subsetting rows: 
# get first 5 rows
df1[:5]  # option 1 - not recommended bcoz not clear if the args are labels or integer locations 
df1.loc[:'a5']  # option 2  # .loc works based on label, not numeric position 
df1.iloc[:5]  # option 3  # .iloc uses integer positions 



# Subsetting cols: 
x = df1['col_1']
type(x)  # pd Series 

y = df1.col_1  # selecting col using dot notation 
type(y)  # pd Series 

df1[['col_1','col_2']]


#%%Plots 

df1.plot.line()  # note that we need the parens at the end here 

plt.plot(['col_1'], ['col_2'], 'ro--')

plt.show
