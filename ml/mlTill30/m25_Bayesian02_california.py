from xgboost import XGBRegressor, train
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np


x,y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, random_state=42)


# 평가 함수 정의
def xgb_cv(learning_rate, max_depth, min_child_weight,
           subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha, num_leaves, min_child_samples):

    # model = XGBRegressor(
    #     **bayesian_params,
    #     # n_estimators=100,
    #     # random_state=42,
    #     # verbosity=1,
    # )
    model = XGBRegressor(
    learning_rate=learning_rate,
    max_depth=int(max_depth),
    min_child_weight=min_child_weight,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    max_bin=int(max_bin),
    reg_lambda=reg_lambda,
    reg_alpha=reg_alpha,
    n_estimators=100,
    random_state=42,
    verbosity=0,
)

    score = cross_val_score(model, x_train, y_train, cv=15, scoring='neg_root_mean_squared_error')
    return score.mean()  # 보통 음수로 나오므로, 값이 클수록 좋음


bayesian_params = {
    'learning_rate': (0.001, 0.1),
    'max_depth': (3,10),
    'num_leaves': (24,40),
    'min_child_samples': (10,200),
    'min_child_weight': (1,50),
    'subsample': (0.5,1),
    'colsample_bytree': (0.5,1),
    'max_bin': (9,500),
    'reg_lambda': (0,10),
    'reg_alpha': (0.01,50),
}


from bayes_opt import BayesianOptimization
optimizer = BayesianOptimization(
    f = xgb_cv,         #블랙박스 함수 #얘를 xgboost
    pbounds=bayesian_params,
    random_state=42,
    verbose=1,
)


optimizer.maximize(init_points= 5,
                   n_iter= 30)

print('best parameter!!')
print(optimizer.max)

