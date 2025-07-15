# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_california_housing, load_diabetes
# import pandas as pd

# import os

# if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
#     BASE_PATH = '/workspace/TensorJae/Study25/'
# else:                                                 # 로컬인 경우
#     BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
# basepath = os.path.join(BASE_PATH)

# #1 data


# path = basepath + '_data/dacon/ddarung/'

# train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
# # print(train_csv) #[1459 rows x 10 columns]

# test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
# # print(test_csv) #[715 rows x 9 columns]

# submission_csv = pd.read_csv(path+ 'submission.csv', index_col=0)
# # print(submission_csv) # [715 rows x 1 columns]

# train_csv = train_csv.dropna()


# # test_csv = test_csv.dropna()
# test_csv = test_csv.fillna(train_csv.mean())

# x = train_csv.drop(['count'], axis=1)  #count라는 axis=1 열 삭제, 행은 axis =0
# feature_names = x.columns
# print(x) #[1459 rows x 9 columns]

# y = train_csv['count'] 
# print(y) #(1459,)   
# # (442,)
# # 25.0 346.0
# # exit()

# log_y = np.log1p(y)  # log(1 + y) 변환 (0 대비 안전함)

# # 3. 히스토그램 시각화
# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# plt.hist(y, bins=50, color='blue', alpha=0.5)
# plt.title('Original Target (House Value)')

# plt.subplot(1, 2, 2)
# plt.hist(log_y, bins=50, color='red', alpha=0.5)
# plt.title('Log Transformed Target')

# plt.tight_layout()
# plt.show()





import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd

import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

#1 data


path = basepath + '_data/dacon/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
# print(train_csv) #[1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
# print(test_csv) #[715 rows x 9 columns]

submission_csv = pd.read_csv(path+ 'submission.csv', index_col=0)
# print(submission_csv) # [715 rows x 1 columns]

train_csv = train_csv.dropna()


# test_csv = test_csv.dropna()
test_csv = test_csv.fillna(train_csv.mean())

x = train_csv.drop(['count'], axis=1)  #count라는 axis=1 열 삭제, 행은 axis =0
feature_names = x.columns
print(x) #[1459 rows x 9 columns]

y = train_csv['count'] 
print(y) #(1459,)   
# (442,)
# 25.0 346.0
# exit()
# 2. 로그 변환
# x는 음수값이 있어 NaN 발생 가능 → clip 후 처리
x_clipped = np.clip(x, a_min=0, a_max=None)  # 음수 0으로
log_x = np.log1p(x_clipped)
log_y = np.log1p(y)  # y는 항상 양수

# # 3. NaN 보정 (혹시라도)
# log_x = SimpleImputer(strategy='constant', fill_value=0).fit_transform(log_x)

# 4. 모델 정의
model = LinearRegression()

# 5. 모델 평가 함수
def evaluate_model(x_data, y_data, description):
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # 로그 y를 예측한 경우 → 역변환
    if "로그 y" in description:
        y_test = np.expm1(y_test)
        y_pred = np.expm1(y_pred)

    score = r2_score(y_test, y_pred)
    print(f"{description:<30} → R² Score: {score:.4f}")

# 6. 네 가지 케이스 출력
print("\n✅ 모델 성능 비교 (R² Score)")
evaluate_model(x, y, "1. 원본 x, 원본 y")
evaluate_model(x, log_y, "2. 원본 x, 로그 y")
evaluate_model(log_x, y, "3. 로그 x, 원본 y")
evaluate_model(log_x, log_y, "4. 로그 x, 로그 y")

# # 3. 히스토그램 시각화
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(y, bins=50, color='blue', alpha=0.5)
plt.title('Original Target (House Value)')

plt.subplot(1, 2, 2)
plt.hist(log_y, bins=50, color='red', alpha=0.5)
plt.title('Log Transformed Target')

plt.tight_layout()
plt.show()




# ✅ 모델 성능 비교 (R² Score)
# 1. 원본 x, 원본 y                  → R² Score: 0.6046
# 2. 원본 x, 로그 y                  → R² Score: 0.5652
# 3. 로그 x, 원본 y                  → R² Score: 0.5548
# 4. 로그 x, 로그 y                  → R² Score: 0.5783





from sklearn.datasets import fetch_california_housing

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

import random
seed = 333
random.seed(seed)
np.random.seed(seed)

from sklearn.ensemble import BaggingRegressor, BaggingClassifier, VotingRegressor, VotingClassifier



import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

#1 data


path = basepath + '_data/dacon/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
# print(train_csv) #[1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
# print(test_csv) #[715 rows x 9 columns]

submission_csv = pd.read_csv(path+ 'submission.csv', index_col=0)
# print(submission_csv) # [715 rows x 1 columns]

train_csv = train_csv.dropna()


# test_csv = test_csv.dropna()
test_csv = test_csv.fillna(train_csv.mean())

x = train_csv.drop(['count'], axis=1)  #count라는 axis=1 열 삭제, 행은 axis =0
feature_names = x.columns
print(x) #[1459 rows x 9 columns]

y = train_csv['count'] 

# ========================= log 변환 ===========================
# y = np.log1p(y)
x = np.log1p(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=190, shuffle=True, 
    # stratify=y,
)

xgb = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor()

model = VotingRegressor(
    estimators=[('XGB', xgb), ('LG', lg), ('CAT', cat)],
    # voting='soft' # 디폴트는 hard
)

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('최종점수 : ', results)