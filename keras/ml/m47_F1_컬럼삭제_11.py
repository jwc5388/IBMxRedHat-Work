from sklearn.datasets import load_iris, fetch_covtype, load_digits
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
from sklearn.datasets import load_wine


# 1. Load digits dataset
dataset = load_digits()
x = dataset.data  # shape: (1797, 64)
y = dataset.target  # shape: (1797,)


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
print(type(percentile))

print(xgb.__version__) #내껀 1.7.6 다른 2.1.4


col_name = []
#삭제할 컬럼(25% 이하인 놈들) 을 찾아내자!!
for i, fi in enumerate(model.feature_importances_):
    # print(i, fi)
    if fi <= percentile:
        col_name.append(dataset.feature_names[i])
    else:
        continue
    
print(col_name) 

x = pd.DataFrame(x, columns = dataset.feature_names)
x = x.drop(columns = col_name)



x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed, stratify=y)

model.fit(x_train, y_train)
print('acc2:', model.score(x_test, y_test))    

# =============== XGBClassifier ======================
# acc: 0.9611111111111111
# 25%지점: 0.005079463706351817
# <class 'numpy.float64'>
# 1.7.6
# ['pixel_0_0', 'pixel_1_0', 'pixel_2_0', 'pixel_2_6', 'pixel_2_7', 'pixel_3_0', 'pixel_3_7', 'pixel_4_0', 'pixel_4_7', 'pixel_5_0', 'pixel_5_7', 'pixel_6_0', 'pixel_6_2', 'pixel_6_7', 'pixel_7_0', 'pixel_7_3']
# acc2: 0.9638888888888889