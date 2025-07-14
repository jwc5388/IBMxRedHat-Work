from keras.datasets import mnist, cifar10
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

import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

#1 data

#1 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts=True))   
    
    
    
x_train = x_train/255.
x_test = x_test/255.



x = np.concatenate([x_train, x_test], axis=0)

print(x.shape) #(60000, 32, 32, 3)

x = x.reshape(60000,32*32*3)

# exit()


pca = PCA(n_components=32*32*3)
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)

print(evr_cumsum)

print('0.95 이상 :', np.argmax(evr_cumsum>= 0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum>= 0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum>= 0.999)+1)
if np.any(evr_cumsum >= 1.0):
    print('1.0 이상 :', np.argmax(evr_cumsum >= 1.0) + 1)
else:
    print('1.0 이상을 만족하는 주성분 수 없음')
    



# 0.95 이상 : 217
# 0.99 이상 : 660
# 0.999 이상 : 1433
# 1.0 이상 : 3072