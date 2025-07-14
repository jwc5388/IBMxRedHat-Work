# kaggle bank

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import pandas as pd

from sklearn.utils import all_estimators
import sklearn as sk
print(sk.__version__)  # 1.6.1

# 1. Load Data
path = './Study25/_data/kaggle/bank/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. Encode categorical features
le_geo = LabelEncoder()
le_gender = LabelEncoder()

train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])

test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

# 3. Drop unneeded columns
train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# 4. Separate features and target
x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']


n_split = 5

# kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)


#회귀는 stratfied 안된다
kfold = StratifiedKFold(n_splits=n_split, shuffle= True, random_state=42)



# 회귀 모델 전부 가져오기
allAlgorithms = all_estimators(type_filter= 'classifier')
# 결과 저장용
max_score = 0
max_name = None
model_scores = []

# 모델 반복 평가
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        scores = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')  # ✅ 정확도 기준
        mean_score = np.mean(scores)
        model_scores.append((name, mean_score))
        
        if mean_score > max_score:
            max_score = mean_score
            max_name = name
        
        print(f'{name:<40} 정확도 평균 점수: {mean_score:.4f}')
        
    except Exception as e:
        print(f'{name:<40} ⚠️ 에러 발생')

# 최고 성능 모델 출력
model_scores.sort(key=lambda x: x[1], reverse=True)
print('\n✅ 최고 성능 모델:', model_scores[0])


# AdaBoostClassifier                       정확도 평균 점수: 0.8576
# BaggingClassifier                        정확도 평균 점수: 0.8483
# BernoulliNB                              정확도 평균 점수: 0.7924
# CalibratedClassifierCV                   정확도 평균 점수: 0.7859
# CategoricalNB                            정확도 평균 점수: 0.8259
# ClassifierChain                          ⚠️ 에러 발생
# ComplementNB                             정확도 평균 점수: 0.5893
# DecisionTreeClassifier                   정확도 평균 점수: 0.7983
# DummyClassifier                          정확도 평균 점수: 0.7884
# ExtraTreeClassifier                      정확도 평균 점수: 0.7977
# ExtraTreesClassifier                     정확도 평균 점수: 0.8553
# FixedThresholdClassifier                 ⚠️ 에러 발생
# GaussianNB                               정확도 평균 점수: 0.7924