from sklearn.datasets import load_iris, fetch_california_housing, fetch_covtype
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier, XGBRegressor 
import random
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.decomposition import PCA
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



#1 data
dataset = fetch_covtype()
x = dataset.data
y = dataset.target


# ✅ LabelEncoder 적용
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=42, train_size=0.8, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier(random_state = seed)
model.fit(x_train, y_train)
print("===============", model.__class__.__name__, "======================")
print('acc:', model.score(x_test, y_test))   
print(model.feature_importances_)


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



x_f = pd.DataFrame(x, columns = dataset.feature_names)
x1 = x_f.drop(columns=col_name)
x2 = x_f[['Slope', 'Vertical_Distance_To_Hydrology', 'Hillshade_3pm', 'Soil_Type_0', 'Soil_Type_4', 
          'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_17', 'Soil_Type_18', 
          'Soil_Type_27', 'Soil_Type_33', 'Soil_Type_35']]



x1_train, x1_test, x2_train, x2_test = train_test_split(x1, x2, train_size=0.8, random_state=seed)

print(x1_train.shape, x1_test.shape)
print(x2_train.shape, x2_test.shape)
print(y_train.shape, y_test.shape)

pca = PCA(n_components=1)

x2_train = pca.fit_transform(x2_train)
x2_test = pca.transform(x2_test)

print(x2_train.shape, x2_test.shape)

x_train = np.concatenate([x1_train, x2_train], axis=1)
x_test = np.concatenate([x1_test, x2_test], axis=1)
print(x_train.shape, x_test.shape) 


model.fit(x_train, y_train)
print('FI_Drop + PCA:', model.score(x_test, y_test))      

# FI_Drop + PCA: 0.48540915466898443