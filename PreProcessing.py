# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 08:42:31 2021

@author: FARZAN
"""

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
#%%
dataset = pd.read_csv("DataSet.csv")
#%%
X=dataset.iloc[:,0:3].values
X1=dataset.iloc[:,0:3].values
X2=dataset.iloc[:,0:3].values

Y=dataset.iloc[:,3]

#%%
from sklearn.impute import SimpleImputer
#%%
imputer= SimpleImputer(missing_values=np.nan, strategy="mean")

imputer=imputer.fit(X[:,1:3])             

X[:,1:3]= imputer.transform(X[:,1:3])
#%%
imputer= SimpleImputer(missing_values=np.nan, strategy="most_frequent")

imputer=imputer.fit(X[:,1:3])             

X1[:,1:3]= imputer.transform(X[:,1:3])
#%%
imputer= SimpleImputer(missing_values=np.nan, strategy="constant")

imputer=imputer.fit(X[:,1:3])             

X2[:,1:3]= imputer.transform(X[:,1:3])
#%%
imputer= SimpleImputer(missing_values=np.nan, strategy="median")

imputer=imputer.fit(Y[:,])             

Y[:,]= imputer.transform(Y[:,])
#%%

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X[:,0]=labelencoder_X.fit_transform(X[:,0])

onehotencoder=OneHotEncoder(categorical_features = [0])

X=onehotencoder.fit_transform(X).toarray()
#%%
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

X=sc_X.fit_transform(X)