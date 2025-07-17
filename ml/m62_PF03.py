
import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler


from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#1 data
dataset = load_diabetes()
x = dataset.data
y = dataset.target


pf = PolynomialFeatures(degree=2, include_bias=False)

x_pf = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_pf, y, train_size=0.8, random_state=seed, shuffle=True, 
    # stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = XGBRegressor()

model.fit(x_train,y_train)

# model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('최종점수 : ', results)


# print(x.shape, y.shape)
# print(np.unique(y, return_counts=True))       
# print(pd.value_counts(y))           


# 최종점수 :  0.3577208977515455