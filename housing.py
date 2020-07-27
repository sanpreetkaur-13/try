import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_csv('desktop/dataset/housing.csv')

dataset.isnull().sum()

from sklearn.impute import SimpleImputer
sim= SimpleImputer(missing_values=np.nan,strategy='median')
dataset[['total_bedrooms']]=sim.fit_transform(dataset[['total_bedrooms']])

X=dataset.iloc[:,[0,1,2,3,4,5,6,7,9]]
y=dataset.iloc[:,8]
plt.scatter(X['housing_median_age'],y)
plt.show()


pd.plotting.scatter_matrix(dataset)
