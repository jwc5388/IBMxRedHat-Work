# californiaimport numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.utils import all_estimators
import sklearn as sk
print(sk.__version__)  # 1.6.1

x,y = fetch_california_housing(return_X_y=True)

# KFold 설정
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 스케일링
scaler = RobustScaler()
x = scaler.fit_transform(x)

# 회귀 모델 전부 가져오기
allAlgorithms = all_estimators(type_filter='regressor')

# 결과 저장 변수
max_score = 0
max_name = None
model_scores = []

# 모델별 KFold 교차검증 평가
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        scores = cross_val_score(model, x, y, cv=kfold, scoring='r2')
        mean_score = np.mean(scores)
        model_scores.append((name, mean_score))
        
        if mean_score > max_score:
            max_score = mean_score
            max_name = name
        
        print(f'{name:<40} R² 평균 점수: {mean_score:.4f}')
        
    except Exception as e:
        print(f'{name:<40} ⚠️ 에러 발생')

# 최고 성능 모델 출력
model_scores.sort(key=lambda x: x[1], reverse=True)
print('\n✅ 최고 성능 모델:', model_scores[0])


# ARDRegression                            R² 평균 점수: 0.6025
# AdaBoostRegressor                        R² 평균 점수: 0.3948
# BaggingRegressor                         R² 평균 점수: 0.7925
# BayesianRidge                            R² 평균 점수: 0.6014
# CCA                                      ⚠️ 에러 발생
# DecisionTreeRegressor                    R² 평균 점수: 0.6176
# DummyRegressor                           R² 평균 점수: -0.0003
# ElasticNet                               R² 평균 점수: 0.1448
# ElasticNetCV                             R² 평균 점수: 0.6012
# ExtraTreeRegressor                       R² 평균 점수: 0.5818
# ExtraTreesRegressor                      R² 평균 점수: 0.8152
# GammaRegressor                           R² 평균 점수: 0.3327
# GaussianProcessRegressor                 R² 평균 점수: -117.4647
# GradientBoostingRegressor                R² 평균 점수: 0.7877
# HistGradientBoostingRegressor            R² 평균 점수: 0.8349
# HuberRegressor                           R² 평균 점수: -1.5309
# IsotonicRegression                       ⚠️ 에러 발생
# KNeighborsRegressor                      R² 평균 점수: 0.6879
# KernelRidge                              R² 평균 점수: -1.1522
# Lars                                     R² 평균 점수: 0.6014
# LarsCV                                   R² 평균 점수: 0.6012
# Lasso                                    R² 평균 점수: -0.0003
# LassoCV                                  R² 평균 점수: 0.6014
# LassoLars                                R² 평균 점수: -0.0003
# LassoLarsCV                              R² 평균 점수: 0.6014
# LassoLarsIC                              R² 평균 점수: 0.6013
# LinearRegression                         R² 평균 점수: 0.6014
# LinearSVR                                R² 평균 점수: -2.1652
# MLPRegressor                             R² 평균 점수: 0.7003
# MultiOutputRegressor                     ⚠️ 에러 발생
# MultiTaskElasticNet                      ⚠️ 에러 발생
# MultiTaskElasticNetCV                    ⚠️ 에러 발생
# MultiTaskLasso                           ⚠️ 에러 발생
# MultiTaskLassoCV                         ⚠️ 에러 발생
# NuSVR                                    R² 평균 점수: 0.6875
# OrthogonalMatchingPursuit                R² 평균 점수: 0.4732
# OrthogonalMatchingPursuitCV              R² 평균 점수: 0.5346
# PLSCanonical                             ⚠️ 에러 발생
# PLSRegression                            R² 평균 점수: 0.5226
# PassiveAggressiveRegressor               R² 평균 점수: -24.8619
# PoissonRegressor                         R² 평균 점수: 0.4196
# QuantileRegressor                        R² 평균 점수: -0.0555
# RANSACRegressor                          R² 평균 점수: -0.7400
# RadiusNeighborsRegressor                 R² 평균 점수: nan
# RandomForestRegressor                    R² 평균 점수: 0.8103
# RegressorChain                           ⚠️ 에러 발생
# Ridge                                    R² 평균 점수: 0.6014
# RidgeCV                                  R² 평균 점수: 0.6014
# SGDRegressor                             R² 평균 점수: -243143100136668337799168.0000
# SVR                                      R² 평균 점수: 0.6847
# StackingRegressor                        ⚠️ 에러 발생
# TheilSenRegressor                        R² 평균 점수: -11.1268
# TransformedTargetRegressor               R² 평균 점수: 0.6014
# TweedieRegressor                         R² 평균 점수: 0.3394
# VotingRegressor                          ⚠️ 에러 발생

# ✅ 최고 성능 모델: ('HistGradientBoostingRegressor', 0.8349012955295272)