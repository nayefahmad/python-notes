# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 02:58:22 2019

@author: Nayef
"""

import numpy as np
import pandas as pd
import statsmodels as sm 
import os 

os.getcwd()

df1 = pd.read_csv("D:/pydata-book/datasets/titanic/train.csv")

df1.shape
df1.columns 
df1.head
df1.tail

df1.plot(kind = "scatter", x = "Age", y = "Fare")

