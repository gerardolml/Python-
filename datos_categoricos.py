#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 18:58:27 2020

@author: gerardo
"""

# Datos categoricos 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# Codificar datos categoricos 

from sklearn import preprocessing
le_x = preprocessing.LabelEncoder()
x[:,0]= le_x.fit_transform(x[:,0])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
 
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
x = np.array(ct.fit_transform(x), dtype=np.float)

le_y = preprocessing.LabelEncoder()
y = le_y.fit_transform(y)
