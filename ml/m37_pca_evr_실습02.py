from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import sklearn as sk
import pandas as pd
import numpy as np
from tabnanny import verbose
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score, train_test_split
import warnings
from bayes_opt import BayesianOptimization
import random
import time
from keras.callbacks import EarlyStopping
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor, BaggingClassifier, RandomForestRegressor

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

# 데이터 로드 및 전처리
x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x = np.concatenate([x_train, x_test], axis=0)

print(x.shape) #(20640, 8)

# exit()


pca = PCA(n_components=8)
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)

# print(evr_cumsum)




print('0.95 이상 :', np.argmax(evr_cumsum>= 0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum>= 0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum>= 0.999)+1)
if np.any(evr_cumsum >= 1.0):
    print('1.0 이상 :', np.argmax(evr_cumsum >= 1.0) + 1)
else:
    print('1.0 이상을 만족하는 주성분 수 없음')


# 0.95 이상 : 3
# 0.99 이상 : 4
# 0.999 이상 : 6
# 1.0 이상을 만족하는 주성분 수 없음

    

# #1 1.0일떄 몇개?
# # 0.999 이상 몇개?
# # 0.99 이상 몇개?
# # 0.95 이상 몇개?

# count1 = 0
# count2 = 0
# count3 = 0
# count4 =0

# print(evr_cumsum.shape)

# # exit()
# for i in range(len(evr_cumsum)):
#     if evr_cumsum[i]==1.0:
#         count1 += 1
#     if evr_cumsum[i]>=0.999:
#         count2 += 1
#     if evr_cumsum[i]>=0.99:
#         count3 +=1
#     if evr_cumsum[i]>= 0.95:
#         count4 +=1
        
# print(count1)
# print(count2)
# print(count3)
# print(count4)