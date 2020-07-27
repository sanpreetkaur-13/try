import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_csv('desktop/bank-additional.csv', na_values ='?' , sep=';')

dataset.isnull().sum()

y=dataset.iloc[:,-1]
X=dataset.iloc[:,0:20]


X_int=X.iloc[:,[0,10,11,12,13,15,16,17,18,19]]
X_str=X.iloc[:,[1,2,3,4,5,6,7,8,9,14]]

X_str=pd.get_dummies(X_str)

X_int=X_int.values
X_str=X_str.values

X_final=np.concatenate([X_int,X_str],axis=1)

from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()

y=lab.fit_transform(y)

lab.classes_
