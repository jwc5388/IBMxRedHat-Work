
import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_wine, fetch_covtype, fetch_california_housing, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
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

# 1. Load Data
path = basepath + '_data/kaggle/bank/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. Encode categorical features
le_geo = LabelEncoder()
le_gender = LabelEncoder()

train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])

test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

# 3. Drop unneeded columns
train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# 4. Separate features and target
x = train_csv.drop(['Exited'], axis=1)
feature_names = x.columns
y = train_csv['Exited']

pf = PolynomialFeatures(degree=2, include_bias=False)

x_pf = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_pf, y, train_size=0.8, random_state=seed, shuffle=True, 
    # stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = XGBClassifier()

model.fit(x_train,y_train)

# model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('최종점수 : ', results)


# print(x.shape, y.shape)
# print(np.unique(y, return_counts=True))       
# print(pd.value_counts(y))           

# 최종점수 :  0.8640591389705214