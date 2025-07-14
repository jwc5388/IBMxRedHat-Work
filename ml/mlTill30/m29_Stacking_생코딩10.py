import numpy as np
from sklearn.datasets import fetch_california_housing, load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

#1 data

#1 data
x,y = fetch_covtype(return_X_y=True)


# ✅ LabelEncoder 적용
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=42, train_size=0.8, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



xgb = XGBClassifier()
rf = RandomForestClassifier()
# cat = CatBoostClassifier(verbose=0)
lg = LGBMClassifier(verbose=0)


models = [xgb,rf, lg]

train_list = []
test_list = []

for model in models:
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    train_list.append(y_train_pred)
    test_list.append(y_test_pred)
    
    score = accuracy_score(y_test, y_test_pred)
    class_name = model.__class__.__name__
    print('{0} ACC : {1:.4f}'.format(class_name, score))
    
    
x_train_new = np.array(train_list).T
# x_train_new = x_train_new.T
print(x_train_new)
print(x_train_new.shape)


x_test_new = np.array(test_list).T


#2-2



model2 = XGBClassifier(verbose=0)
model2.fit(x_train_new, y_train)

y_pred2 = model2.predict(x_test_new)
score2 = accuracy_score(y_test, y_pred2)
print(
    '{0} stacking result:'.format(model2.__class__.__name__), score2
)


# stacking result: 0.9912280701754386