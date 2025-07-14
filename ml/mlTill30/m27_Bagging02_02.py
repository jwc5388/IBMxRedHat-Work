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

#2 model

# model = DecisionTreeRegressor()
model = BaggingRegressor(DecisionTreeRegressor(), n_estimators=100, 
                         n_jobs=-2, 
                         random_state=42, 
                        #  bootstrap=True,)           #false 성능 개 구림
)

# model = RandomForestRegressor(random_state=42)
#3 train
model.fit(x_train, y_train)


#4 train, predict
result = model.score(x_test, y_test)
print('final score:', result)

#decision tree
# final score: 0.6183215430363695


#bagging 
# final score: 0.8042405904305805 #샘플데이터 중복허용



#bagging/ bootstrap = false
# final score: 0.6456814532339009   #샘플데이터 중복허용 x


#randomforestregressor
# final score: 0.8043856726253765v