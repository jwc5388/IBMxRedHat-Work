from sklearn.datasets import load_iris, fetch_california_housing
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

datasets = fetch_california_housing()
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


# acc: 0.8331889210779453
# [0.44076005 0.07022501 0.04738963 0.02539878 0.02791631 0.15494779
#  0.10437357 0.12898892]
# 25%지점: 0.0425213030539453
# <class 'numpy.float64'>
# 1.7.6
# ['AveBedrms', 'Population']
# r2: 0.8375562606091514