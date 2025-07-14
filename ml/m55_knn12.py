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

# 1. 데이터 로드
path = basepath + '_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv')
test_csv = pd.read_csv(path + 'test.csv')
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 2. 피처/타겟 분리 및 기본 피처 리스트 생성
y = train_csv['target']
x = train_csv.drop(['ID_code', 'target'], axis=1)
x_submit = test_csv.drop(['ID_code'], axis=1)
original_features = [col for col in x.columns]

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)


y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('accuracy score:', acc)

print('f1 score:', f1_score(y_test, y_pred))



# f1 score: 0.0009928021841648052

