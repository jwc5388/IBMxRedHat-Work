from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
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

#1 data

# === Load Data ===
path = basepath + '_data/dacon/cancer/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# === Separate Features and Target Label ===
x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']


# === One-Hot Encode Categorical Columns ===
categorical_cols = x.select_dtypes(include='object').columns
x = pd.get_dummies(x, columns=categorical_cols)
test_csv = pd.get_dummies(test_csv, columns=categorical_cols)
x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)

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
# acc: 0.8794171638366223
# 25%지점: 0.00946331094019115
# <class 'numpy.float64'>
# 1.7.6
# ['Gender_M', 'Country_USA', 'Race_HSP', 'Family_Background_Positive', 'Radiation_History_Unexposed', 'Iodine_Deficiency_Sufficient', 'Smoke_Smoker', 'Weight_Risk_Obese', 'Diabetes_Yes']
# acc2: 0.8825149150986691 


