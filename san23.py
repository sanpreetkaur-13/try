import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Desktop/dataset/data_pre.csv')

X=dataset.iloc[:,0:3].values
y=dataset.iloc[:,-1].values

#yaha toh missing value fix kar di
from sklearn.impute import SimpleImputer
sim=SimpleImputer(missing_values=np.nan,strategy='median')
X[:,0:2]=sim.fit_transform(X[:,0:2])

#Label Encoding
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
X[:,2]=lab.fit_transform(X[:,2]) 
y=lab.fit_transform(y)

from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()
X_min_max=minmax.fit_transform(X)
