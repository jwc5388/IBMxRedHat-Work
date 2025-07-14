from sklearn.datasets import load_breast_cancer, fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
import random
import numpy as np
import xgboost as xgb
import pandas as pd
#1data

seed = 123

random.seed(seed)
np.random.seed(seed)

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

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed, )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = KNeighborsRegressor(n_neighbors=5)

model.fit(x_train, y_train)


y_pred = model.predict(x_test)
# acc = accuracy_score(y_test, y_pred)
# print('accuracy score:', acc)

print("===============", model.__class__.__name__, "======================")

print('r2 score:', r2_score(y_test, y_pred))


# r2 score: 0.7019773541270863
