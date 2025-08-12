# 이진cancer dacondiab kagglebank 다중wine fetchcov digits

#데이터셋, 스케일러, 모델 다같이

#[결과] 6개의 데이터셋에서 어떤 스케일러와 어떤 모델을 썼을때 성능이 얼마야
# 라고 출력시켜라. 스케일러 모델 - pipeline

#여섯개 데이터 , 스케일러, 모델 5,4 for 문으로


import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer, fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)
def load_dacon_diabetes(basepath):
    path = os.path.join(basepath, "_data/diabetes/")
    train_csv = pd.read_csv(os.path.join(path, "train.csv"), index_col=0)

    X = train_csv.drop(columns=["Outcome"])
    y = train_csv["Outcome"].values

    # 0값을 결측으로 보고 평균 대체
    X = X.replace(0, np.nan)
    X = X.fillna(train_csv.mean(numeric_only=True))

    return X.values, y
# 2 데이터셋



datasets = [
    ("Iris", load_iris(return_X_y=True)),
    ("Wine", load_wine(return_X_y=True)),
    ("Digits", load_digits(return_X_y=True)),
    # ("Covtype", fetch_covtype(return_X_y=True))
    # ("BreastCancer", load_breast_cancer(return_X_y=True)),
    ("DaconDiabetes", load_dacon_diabetes(BASE_PATH)),
]





scalers = [
    ("MinMaxScaler", MinMaxScaler()),
    ("StandardScaler", StandardScaler()),
    ("RobustScaler", RobustScaler()),
    ("MaxAbsScaler", MaxAbsScaler())
]


models = [
    ("XGBoost", XGBClassifier()),
    ("LGBM", LGBMClassifier()),
    ("Cat", CatBoostClassifier()),
    ("SVC", SVC()),
    ("Randomforest", RandomForestClassifier())

]


result = []

from sklearn.preprocessing import LabelEncoder

# results = []

for data_name, (X, y) in datasets:
    # 1) 레이블 0~(K-1)로 강제 인코딩
    le = LabelEncoder()
    y_enc = le.fit_transform(y)          # 예: [1..7] -> [0..6]
    n_classes = np.unique(y_enc).size
    if n_classes < 2:
        continue

    # 2) 항상 stratify=y_enc
    x_train, x_test, y_train, y_test = train_test_split(
        X, y_enc, train_size=0.8, random_state=42, stratify=y_enc
    )

    # 3) XGB를 포함한 모델 생성 (다중/이진 자동 세팅)
    models = [
        ("XGBoost", XGBClassifier(
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            tree_method="hist",
            n_estimators=300,
            random_state=42
        )),
        ("LGBM", LGBMClassifier(
            objective="multiclass" if n_classes > 2 else "binary",
            n_estimators=300,
            random_state=42
        )),
        ("Cat", CatBoostClassifier(
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
            iterations=300, random_seed=42
        )),
        ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=42)),
        ("SVC", SVC())
    ]

    for scaler_name, scaler in scalers:
        for model_name, model in models:
            pipe = make_pipeline(PCA(n_components=4),scaler, model)
            pipe.fit(x_train, y_train)
            score = pipe.score(x_test, y_test)
            result.append([data_name, scaler_name, model_name, round(score, 4)])


results = pd.DataFrame(result, columns=['Dataset', 'Scaler','Model', 'Accuracy'])
print(results)

print('\n Best Combinations')
print(results.sort_values(by ='Accuracy', ascending=False).head(50))

# Best Combinations
#    Dataset          Scaler         Model  Accuracy
# 4     Iris    MinMaxScaler           SVC    1.0000
# 19    Iris    MaxAbsScaler           SVC    1.0000
# 28    Wine  StandardScaler  RandomForest    0.9722
# 33    Wine    RobustScaler  RandomForest    0.9722
# 24    Wine    MinMaxScaler           SVC    0.9722
# 23    Wine    MinMaxScaler  RandomForest    0.9722
# 34    Wine    RobustScaler           SVC    0.9722
# 29    Wine  StandardScaler           SVC    0.9722
# 38    Wine    MaxAbsScaler  RandomForest    0.9722
# 39    Wine    MaxAbsScaler           SVC    0.9722
# 9     Iris  StandardScaler           SVC    0.9667
# 37    Wine    MaxAbsScaler           Cat    0.9444
# 31    Wine    RobustScaler          LGBM    0.9444
# 27    Wine  StandardScaler           Cat    0.9444
# 26    Wine  StandardScaler          LGBM    0.9444
# 25    Wine  StandardScaler       XGBoost    0.9444
# 22    Wine    MinMaxScaler           Cat    0.9444
# 20    Wine    MinMaxScaler       XGBoost    0.9444
# 32    Wine    RobustScaler           Cat    0.9444
# 30    Wine    RobustScaler       XGBoost    0.9444
# 35    Wine    MaxAbsScaler       XGBoost    0.9444
# 36    Wine    MaxAbsScaler          LGBM    0.9444
# 17    Iris    MaxAbsScaler           Cat    0.9333
# 14    Iris    RobustScaler           SVC    0.9333
# 2     Iris    MinMaxScaler           Cat    0.9333
# 12    Iris    RobustScaler           Cat    0.9333
# 11    Iris    RobustScaler          LGBM    0.9333
# 7     Iris  StandardScaler           Cat    0.9333
# 21    Wine    MinMaxScaler          LGBM    0.9167
# 1     Iris    MinMaxScaler          LGBM    0.9000
# 0     Iris    MinMaxScaler       XGBoost    0.9000
# 3     Iris    MinMaxScaler  RandomForest    0.9000
# 18    Iris    MaxAbsScaler  RandomForest    0.9000
# 16    Iris    MaxAbsScaler          LGBM    0.9000
# 15    Iris    MaxAbsScaler       XGBoost    0.9000
# 13    Iris    RobustScaler  RandomForest    0.9000
# 10    Iris    RobustScaler       XGBoost    0.9000
# 8     Iris  StandardScaler  RandomForest    0.9000
# 6     Iris  StandardScaler          LGBM    0.9000
# 5     Iris  StandardScaler       XGBoost    0.9000
# 59  Digits    MaxAbsScaler           SVC    0.8694
# 44  Digits    MinMaxScaler           SVC    0.8694
# 49  Digits  StandardScaler           SVC    0.8667
# 54  Digits    RobustScaler           SVC    0.8639
# 42  Digits    MinMaxScaler           Cat    0.8556
# 47  Digits  StandardScaler           Cat    0.8556
# 52  Digits    RobustScaler           Cat    0.8556
# 57  Digits    MaxAbsScaler           Cat    0.8556
# 58  Digits    MaxAbsScaler  RandomForest    0.8472
# 53  Digits    RobustScaler  RandomForest    0.8472
# (base) jaewoo000@JaeBook-Pro IBM:RedHat % 