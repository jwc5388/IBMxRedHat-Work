import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=42)

xgb = XGBRegressor()
# rf = RandomForestRegressor()
cat = CatBoostRegressor(verbose=0)
lg = LGBMRegressor(verbose=0)


models = [xgb, cat, lg]

train_list = []
test_list = []

for model in models:
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    train_list.append(y_train_pred)
    test_list.append(y_test_pred)
    
    score = r2_score(y_test, y_test_pred)
    class_name = model.__class__.__name__
    print('{0} R2 : {1:.4f}'.format(class_name, score))
    
    
x_train_new = np.array(train_list)
x_train_new = x_train_new.T
print(x_train_new)
print(x_train_new.shape)


x_test_new = np.array(test_list).T


#2-2


# XGBRegressor R2 : 0.8286
# RandomForestRegressor R2 : 0.8074
# CatBoostRegressor R2 : 0.8492
# LGBMRegressor R2 : 0.8360

# [[1.24695408 1.10867    1.19408902 1.36973594]
#  [3.56740522 3.7429405  3.39126347 3.58120416]
#  [1.77422929 1.9073003  1.89801405 1.92000068]
#  ...
#  [2.15286732 2.1214     2.21515912 1.91360487]
#  [2.65715718 2.75788    2.78862732 2.63370525]
#  [3.10334921 3.2289602  3.32597659 3.38498271]]
# (16512, 4)


model2 = LGBMRegressor(verbose=0)
model2.fit(x_train_new, y_train)

y_pred2 = model2.predict(x_test_new)
score2 = r2_score(y_test, y_pred2)
print(
    'stacking result:', score2
)


# stacking result: 0.7852545694050193


# stacking result: 0.8088125601830417


# stacking result: 0.8100352208702445