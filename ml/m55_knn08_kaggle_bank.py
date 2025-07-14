from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

print(x.shape, y.shape)      
print(np.unique(y, return_counts=True))        
# (165034, 10) (165034,)
# (array([0, 1]), array([130113,  34921]))
# exit()

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42, stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from imblearn.over_sampling import SMOTE
import sklearn as sk

model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)


y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)


print("===============", model.__class__.__name__, "======================")

print('accuracy score:', acc)

print('f1 score:', f1_score(y_test, y_pred))



# accuracy score: 0.732824427480916
# f1 score: 0.5977011494252874


