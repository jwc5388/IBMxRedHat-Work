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
from sklearn.decomposition import PCA
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
print('r2:', model.score(x_test, y_test))   
print(model.feature_importances_)


print('25%지점:', np.percentile(model.feature_importances_, 25)) 

# exit()

percentile = np.percentile(model.feature_importances_, 25)
print(type(percentile))
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


x_f = pd.DataFrame(x, columns = datasets.feature_names)
x1 = x_f.drop(columns=col_name)
x2 = x_f[['age', 'sex', 's1']]





# print(x2)
# x_f = x_f.drop(columns = col_name)

x1_train, x1_test, x2_train, x2_test = train_test_split(x1, x2, train_size=0.8, random_state=seed)

print(x1_train.shape, x1_test.shape)
print(x2_train.shape, x2_test.shape)
print(y_train.shape, y_test.shape)

# (16512, 6) (4128, 6)
# (16512, 2) (4128, 2)
# (16512,) (4128,)


pca = PCA(n_components=1)

x2_train = pca.fit_transform(x2_train)
x2_test = pca.transform(x2_test)

print(x2_train.shape, x2_test.shape)#(16512, 1) (4128, 1)

x_train = np.concatenate([x1_train, x2_train], axis=1)
x_test = np.concatenate([x1_test, x2_test], axis=1)
print(x_train.shape, x_test.shape)  #(16512, 7) (4128, 7)


model.fit(x_train, y_train)
print('FI_Drop + PCA:', model.score(x_test, y_test))      


# FI_Drop + PCA: 0.4535048560155759