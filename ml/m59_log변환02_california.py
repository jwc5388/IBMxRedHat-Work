# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.datasets import fetch_california_housing

# # # 1. 데이터 로딩
# # dataset = fetch_california_housing()
# # x = dataset.data
# # y = dataset.target  # median house value ($100,000 단위)

# # # 2. 로그 변환 (지수분포형 y 대상)
# # print(y)
# # print(y.shape)
# # print(np.min(y), np.max(y))  # 0.14999 ~ 5.00001 등

# # log_y = np.log1p(y)  # log(1 + y) 변환 (0 대비 안전함)

# # # 3. 히스토그램 시각화
# # plt.figure(figsize=(10, 4))

# # plt.subplot(1, 2, 1)
# # plt.hist(y, bins=50, color='blue', alpha=0.5)
# # plt.title('Original Target (House Value)')

# # plt.subplot(1, 2, 2)
# # plt.hist(log_y, bins=50, color='red', alpha=0.5)
# # plt.title('Log Transformed Target')

# # plt.tight_layout()
# # plt.show()


# # #log 변환 전 score
# # #y 만 log 변환 score
# # ### x 만 log 변환 socre
# # ## x,y, log 변환 score 

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_california_housing
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer

# # 1. 데이터 로딩
# dataset = fetch_california_housing()
# x = dataset.data
# y = dataset.target

# # 2. 로그 변환
# # x는 음수값이 있어 NaN 발생 가능 → clip 후 처리
# x_clipped = np.clip(x, a_min=0, a_max=None)  # 음수 0으로
# log_x = np.log1p(x_clipped)
# log_y = np.log1p(y)  # y는 항상 양수

# # # 3. NaN 보정 (혹시라도)
# # log_x = SimpleImputer(strategy='constant', fill_value=0).fit_transform(log_x)

# # 4. 모델 정의
# model = LinearRegression()

# # 5. 모델 평가 함수
# def evaluate_model(x_data, y_data, description):
#     x_train, x_test, y_train, y_test = train_test_split(
#         x_data, y_data, test_size=0.2, random_state=42
#     )
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)

#     # 로그 y를 예측한 경우 → 역변환
#     if "로그 y" in description:
#         y_test = np.expm1(y_test)
#         y_pred = np.expm1(y_pred)

#     score = r2_score(y_test, y_pred)
#     print(f"{description:<30} → R² Score: {score:.4f}")

# # 6. 네 가지 케이스 출력
# print("\n✅ 모델 성능 비교 (R² Score)")
# evaluate_model(x, y, "1. 원본 x, 원본 y")
# evaluate_model(x, log_y, "2. 원본 x, 로그 y")
# evaluate_model(log_x, y, "3. 로그 x, 원본 y")
# evaluate_model(log_x, log_y, "4. 로그 x, 로그 y")



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


x, y = fetch_california_housing(return_X_y=True)

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


# 최고 점수(R2): 0.8358679276132328
# 최적 하이퍼파라미터: {'learning_rate': np.float64(0.0680069886658152), 'max_depth': np.float64(9.492499826709233), 
# 'num_leaves': np.float64(24.532581876056156), 'min_child_samples': np.float64(177.74146321261998), 
# 'min_child_weight': np.float64(17.136580598271063), 'subsample': np.float64(0.6603118204982997), 
# 'colsample_bytree': np.float64(0.8312450521726851), 'max_bin': np.float64(273.3894066462892), 
# 'reg_lambda': np.float64(9.426502212732464), 'reg_alpha': np.float64(0.8783026639713657)}
# Parameters: { "min_child_samples", "num_leaves" } are not used.

#   bst.update(dtrain, iteration=i, fobj=obj)

# 최적 파라미터로 훈련된 최종 모델의 테스트 R2 Score: 0.8514

##################################################### Voting #########################################################################
# 최종점수 :  0.8533001548462601

# 최종점수 :  0.8533971833252891 아무것도 안한거
# 최종점수 :  0.8635356142975465 y만 한거
# 최종점수 :  0.7616655243136835 x만 한거
# 최종점수 :  0.7619681817962484 둘다 한거