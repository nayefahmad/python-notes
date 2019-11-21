# -*- coding: utf-8 -*-
"""
Iris dataset exploration 
2019-11-18
Nayef 
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
from sklearn import datasets


#%% Shortcuts 
# Ctrl + Enter to run only current line 
# Shift + Enter to run current section 





#%% Data import and first look 

# Convert sklearn Bunch object to a Pandas DataFrame: 
# https://stackoverflow.com/questions/38105539/how-to-convert-a-scikit-learn-dataset-to-a-pandas-dataset


iris_dataset = datasets.load_iris()  # from sklearn 

type(iris_dataset)  # sklearn.utils.Bunch
type(iris_dataset['data'])  # numpy.ndarray

# dir() tries to return a list of valid attributes of the object.
# dir is part of the standart python library 
dir(iris_dataset)  
#dir(iris_dataset['data']) # not useful 


# todo: rewrite after going through pandas subsetting tutorials 
iris_dataset['data'][:6,]
iris_dataset['feature_names']
iris_dataset['target']


# np.c_ is the numpy concatenate function
# which is used to concat iris['data'] and iris['target'] arrays 

# Think of it as cbind or bind_cols from R 

df1_iris = pd.DataFrame(data= np.c_[iris_dataset['data'], 
                                 iris_dataset['target']],
                     columns= iris_dataset['feature_names'] + ['target'])

type(df1_iris)
#dir(df1_iris)  # not useful 

df1_iris.info  # not useful
df1_iris.dtypes  # like str
df1_iris.shape  # rows/cols
df1_iris.columns  # names 


df1_iris.head(5)







#%% 
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())