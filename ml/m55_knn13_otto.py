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

# 1. Load Data
path = basepath + '_data/kaggle/otto/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# 2. Feature & Target 분리
x = train_csv.drop(['target'], axis=1)
feature_names = x.columns
y = train_csv['target']



# # ✅ LabelEncoder 적용
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

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
f1 = f1_score(y_test, y_pred, average='macro')
print("F1 Score:", f1)


# =============== KNeighborsClassifier ======================
# accuracy score: 0.7720588235294118
# F1 Score: 0.715966715225559