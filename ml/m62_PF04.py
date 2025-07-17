import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_wine, fetch_covtype, fetch_california_housing
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

import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

#1 data


path = basepath + '_data/dacon/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
# print(train_csv) #[1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
# print(test_csv) #[715 rows x 9 columns]

submission_csv = pd.read_csv(path+ 'submission.csv', index_col=0)
# print(submission_csv) # [715 rows x 1 columns]

train_csv = train_csv.dropna()


# test_csv = test_csv.dropna()
test_csv = test_csv.fillna(train_csv.mean())

x = train_csv.drop(['count'], axis=1)  #count라는 axis=1 열 삭제, 행은 axis =0
feature_names = x.columns
print(x) #[1459 rows x 9 columns]

y = train_csv['count'] 
print(y) #(1459,)    

pf = PolynomialFeatures(degree=3, include_bias=False)

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

# 최종점수 :  0.7807808980678697