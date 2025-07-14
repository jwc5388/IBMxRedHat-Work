from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
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


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42, stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



model = XGBClassifier(random_state = seed)
model.fit(x_train, y_train)
print("===============", model.__class__.__name__, "======================")
print('acc:', model.score(x_test, y_test))    
# print(model.feature_importances_)


print('25%지점:', np.percentile(model.feature_importances_, 25))  



percentile = np.percentile(model.feature_importances_, 25)
# print(type(percentile))

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



x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed, stratify=y)

model.fit(x_train, y_train)
print('acc2:', model.score(x_test, y_test))    

# =============== XGBClassifier ======================
# acc: 0.9129
# 25%지점: 0.003011914435774088
# <class 'numpy.float64'>
# 1.7.6
# ['var_4', 'var_7', 'var_10', 'var_14', 'var_15', 'var_16', 'var_17', 'var_19', 'var_27', 'var_29', 'var_30', 'var_37', 'var_38', 'var_41', 'var_42', 'var_46', 'var_47', 'var_50', 'var_54', 'var_55', 'var_57', 'var_59', 'var_61', 'var_64', 'var_65', 'var_72', 'var_73', 'var_79', 'var_84', 'var_98', 'var_100', 'var_101', 'var_103', 'var_113', 'var_117', 'var_120', 'var_124', 'var_126', 'var_136', 'var_142', 'var_143', 'var_158', 'var_160', 'var_161', 'var_176', 'var_181', 'var_182', 'var_183', 'var_185', 'var_193']
# acc2: 0.91415