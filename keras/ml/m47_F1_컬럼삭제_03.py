from sklearn.datasets import load_iris, fetch_california_housing, load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
import xgboost as xgb
import pandas as pd
#1data

seed = 123

random.seed(seed)
np.random.seed(seed)

datasets = load_diabetes()
x = datasets.data
y= datasets.target
print(x.shape, y.shape)         #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed)


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
        col_name.append(datasets.feature_names[i])
    else:
        continue
    
print(col_name) 

x = pd.DataFrame(x, columns = datasets.feature_names)
x = x.drop(columns = col_name)

# print(x)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed)

model.fit(x_train, y_train)
print('r2:', model.score(x_test, y_test))   


# acc: 0.46031498581302355
# [0.03234755 0.04475458 0.21775821 0.08212134 0.04737134 0.04843808
#  0.06012437 0.09595279 0.30483842 0.06629325]
# 25%지점: 0.04763802606612444
# <class 'numpy.float64'>
# 1.7.6
# ['age', 'sex', 's1']
# r2: 0.37899921143508253