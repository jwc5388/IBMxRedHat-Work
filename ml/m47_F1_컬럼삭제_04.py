from sklearn.datasets import load_iris, fetch_california_housing, load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
import xgboost as xgb
import pandas as pd
#1data

seed = 42

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
print(x) #[1459 rows x 9 columns]

y = train_csv['count'] 
print(y) #(1459,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model = XGBRegressor(random_state = seed)
model.fit(x_train, y_train)
print("===============", model.__class__.__name__, "======================")
print('acc:', model.score(x_test, y_test))     
print(model.feature_importances_)


print('25%지점:', np.percentile(model.feature_importances_, 25))  



percentile = np.percentile(model.feature_importances_, 25)
print(type(percentile))
# for i, fi in enumerate(model.fe)
print(xgb.__version__) #내껀 1.7.6 다른 2.1.4


col_name = []
#삭제할 컬럼(25% 이하인 놈들) 을 찾아내자!!
for i, fi in enumerate(model.feature_importances_):
    # print(i, fi)
    if fi <= percentile:
        col_name.append(x.columns[i])
    else:
        continue
    
print(col_name) 

x = pd.DataFrame(x, columns = x.columns)
x = x.drop(columns = col_name)

# print(x)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed)

model.fit(x_train, y_train)
print('r2:', model.score(x_test, y_test))   


# acc: 0.7925808609925709
# [0.3986371  0.11783976 0.29238915 0.02271906 0.03465068 0.02565262
#  0.03480605 0.04819283 0.02511278]
# 25%지점: 0.02565261535346508
# <class 'numpy.float64'>
# 1.7.6
# ['hour_bef_windspeed', 'hour_bef_visibility', 'hour_bef_pm2.5']
# r2: 0.7897711037028872