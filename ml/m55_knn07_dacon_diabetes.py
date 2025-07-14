from sklearn.datasets import load_breast_cancer
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
path = basepath +  '_data/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col= 0)

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed, stratify=y)



scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)


y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("===============", model.__class__.__name__, "======================")

print('accuracy score:', acc)

print('f1 score:', f1_score(y_test, y_pred))



# =============== KNeighborsClassifier ======================
# accuracy score: 0.732824427480916
# f1 score: 0.5977011494252874
