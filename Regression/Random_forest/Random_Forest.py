#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:05:08 2021

@author: lewisling
"""

# Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv') 
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# # Feature Scaling
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()

# y_train = sc_y.fit_transform(y_train.reshape(-1,1))
# y_train= y_train.reshape(1,-1)[0]


# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0) 
regressor.fit(X, y)


# Predicting a new result 
y_pred = regressor.predict(np.array([[6.5]]))



X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue') 
plt.title('Truth or Bluff (Random Forest Regression)') 
plt.xlabel('Position level')
plt.ylabel('Salary')
