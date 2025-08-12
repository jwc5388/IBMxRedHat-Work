import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing, load_diabetes

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

SEED = 42
np.random.seed(SEED)

# ========= 경로 =========
if os.path.exists('/workspace/TensorJae/Study25/'):
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')

# ========= 1) 데이터 로더들 =========

def load_kaggle_bike(basepath):
    """Kaggle Bike Sharing Demand (회귀)"""
    data_dir = os.path.join(basepath, '_data/kaggle/bike/')
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'), index_col=0)  # index=datetime 문자열
    # datetime 파생
    dt = pd.to_datetime(train.index)
    train = train.copy()
    train['year']      = dt.year
    train['month']     = dt.month
    train['day']       = dt.day
    train['hour']      = dt.hour
    train['dayofweek'] = dt.dayofweek
    train['is_weekend'] = (train['dayofweek'] >= 5).astype(int)

    y = train['count'].astype(float).values
    # 누수 제거
    drop_leak = [c for c in ['count','casual','registered'] if c in train.columns]
    X = train.drop(columns=drop_leak)

    # 범주형 후보
    cat_candidates = ['season','holiday','workingday','weather','year','month','day','hour','dayofweek','is_weekend']
    cat_cols = [c for c in cat_candidates if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # 결측 보정
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

    return X, y, cat_cols, num_cols, 'KaggleBike'

def load_dacon_ddarung(basepath):
    data_dir = os.path.join(basepath, '_data/dacon/ddarung/')
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'), index_col=0)
    # 결측 처리
    train = train.copy()
    train = train.dropna()  # 원 코드 기준 유지

    y = train['count'].astype(float).values
    X = train.drop(columns=['count'])

    # 범주형 자동 감지: 자주 쓰이는 이름 우선
    known_cats = ['season','seasons','holiday','functioning_day']
    cat_cols = [c for c in known_cats if c in X.columns]
    # 타입이 object/범주로 들어온 다른 열도 포함
    cat_cols += [c for c in X.columns if X[c].dtype == 'object' and c not in cat_cols]
    cat_cols = list(dict.fromkeys(cat_cols))  # 중복 제거

    num_cols = [c for c in X.columns if c not in cat_cols]

    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

    return X, y, cat_cols, num_cols, 'DaconDdarung'


def load_california():
    """sklearn California Housing (회귀) — 전부 수치형"""
    X, y = fetch_california_housing(return_X_y=True)
    # numpy 배열에 대해선 컬럼 인덱스로 지정
    cat_cols = []                          # 범주형 없음
    num_cols = list(range(X.shape[1]))     # 모든 열 수치형
    return X, y, cat_cols, num_cols, 'CaliforniaHousing'

def load_diabetes_reg():
    """sklearn Diabetes (회귀) — 전부 수치형"""
    X, y = load_diabetes(return_X_y=True)
    cat_cols = []
    num_cols = list(range(X.shape[1]))
    return X, y, cat_cols, num_cols, 'Diabetes'

datasets = [
    load_kaggle_bike(BASE_PATH),
    load_dacon_ddarung(BASE_PATH),
    load_california(),     # ✅ 추가
    load_diabetes_reg(),   # ✅ 추가
]

# ========= 2) 전처리/모델 조합 =========

scalers = [
    ("MinMax",   MinMaxScaler()),
    ("Standard", StandardScaler()),
    ("Robust",   RobustScaler()),
    ("MaxAbs",   MaxAbsScaler()),
]

models = [
    ("XGB",   XGBRegressor(n_estimators=400, random_state=SEED, tree_method="hist")),
    ("LGBM",  LGBMRegressor(n_estimators=400, random_state=SEED)),
    ("Cat",   CatBoostRegressor(iterations=400, random_seed=SEED, verbose=0)),
    ("RF",    RandomForestRegressor(n_estimators=400, random_state=SEED)),
    ("SVR",   SVR()),
]

# 선형 베이스라인(다항 2차 포함 비교용)
linear_variants = [
    ("Ridge", Ridge(alpha=1.0)),
    ("RidgePoly2",
     Pipeline([
         ("poly", PolynomialFeatures(degree=2, include_bias=False)),
         ("std",  StandardScaler(with_mean=False)),
         ("ridge", Ridge(alpha=1.0))
     ])),
]

USE_LOG_TARGET = True  # 로그타깃 on/off

# ========= 3) 학습 루프 =========

def make_preprocessor(cat_cols, num_cols, scaler):
    transformers = []
    if len(cat_cols) > 0:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    if len(num_cols) > 0:
        transformers.append(("num", Pipeline([("scaler", scaler)]), num_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

def fit_eval(pipe, X_train, X_valid, y_train, y_valid):
    if USE_LOG_TARGET:
        y_tr = np.log1p(y_train)
        y_va = np.log1p(y_valid)
        pipe.fit(X_train, y_tr)
        pred = np.expm1(pipe.predict(X_valid))
    else:
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, pred))
    r2 = r2_score(y_valid, pred)
    return r2, rmse

from sklearn.metrics import mean_squared_error, r2_score

all_results = []

for (X, y, cat_cols, num_cols, dname) in datasets:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, random_state=SEED, shuffle=True
    )

    # 트리/부스팅/커널 + 스케일러 4종
    for sname, scaler in scalers:
        pre = make_preprocessor(cat_cols, num_cols, scaler)
        for mname, model in models:
            pipe = Pipeline([("prep", pre), ("model", model)])
            r2, rmse = fit_eval(pipe, X_train, X_valid, y_train, y_valid)
            all_results.append([dname, f"{sname}+{mname}", r2, rmse])

    # 선형/다항(범주형은 원-핫, 수치는 모델 파이프라인 내부에서 처리)
    ohe_only = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    for lname, lmodel in linear_variants:
        pipe = Pipeline([("ohe", ohe_only), ("model", lmodel)])
        r2, rmse = fit_eval(pipe, X_train, X_valid, y_train, y_valid)
        all_results.append([dname, f"OHE+{lname}", r2, rmse])

# ========= 4) 결과 출력 =========
df = pd.DataFrame(all_results, columns=["Dataset", "Combo", "R2", "RMSE"])

print(df)

print("\n=== Top 20 by R2 ===")
print(df.sort_values(["R2"], ascending=False).head(20))

print("\n=== Top 20 by RMSE (lower is better) ===")
print(df.sort_values(["RMSE"], ascending=True).head(20))


# === Top 20 by R2 ===
#        Dataset           Combo        R2       RMSE
# 4   KaggleBike      MinMax+SVR  0.955123  38.486961
# 19  KaggleBike      MaxAbs+SVR  0.955088  38.502168
# 14  KaggleBike      Robust+SVR  0.950652  40.358650
# 16  KaggleBike     MaxAbs+LGBM  0.950408  40.458514
# 11  KaggleBike     Robust+LGBM  0.950408  40.458514
# 1   KaggleBike     MinMax+LGBM  0.950408  40.458514
# 6   KaggleBike   Standard+LGBM  0.950408  40.458514
# 2   KaggleBike      MinMax+Cat  0.949038  41.013564
# 17  KaggleBike      MaxAbs+Cat  0.949038  41.013564
# 7   KaggleBike    Standard+Cat  0.949038  41.013564
# 12  KaggleBike      Robust+Cat  0.949038  41.013564
# 9   KaggleBike    Standard+SVR  0.945090  42.572363
# 8   KaggleBike     Standard+RF  0.938773  44.954537
# 3   KaggleBike       MinMax+RF  0.938625  45.008765
# 13  KaggleBike       Robust+RF  0.938612  45.013731
# 18  KaggleBike       MaxAbs+RF  0.938584  45.024003
# 21  KaggleBike  OHE+RidgePoly2  0.938121  45.193404
# 15  KaggleBike      MaxAbs+XGB  0.936544  45.765654
# 5   KaggleBike    Standard+XGB  0.935810  46.029557
# 0   KaggleBike      MinMax+XGB  0.933337  46.907782

# === Top 20 by RMSE (lower is better) ===
#               Dataset           Combo        R2      RMSE
# 60  CaliforniaHousing     MaxAbs+LGBM  0.852423  0.439758
# 55  CaliforniaHousing     Robust+LGBM  0.852285  0.439963
# 45  CaliforniaHousing     MinMax+LGBM  0.852115  0.440216
# 50  CaliforniaHousing   Standard+LGBM  0.850671  0.442360
# 56  CaliforniaHousing      Robust+Cat  0.841715  0.455432
# 61  CaliforniaHousing      MaxAbs+Cat  0.841715  0.455432
# 51  CaliforniaHousing    Standard+Cat  0.841715  0.455432
# 46  CaliforniaHousing      MinMax+Cat  0.841715  0.455432
# 49  CaliforniaHousing    Standard+XGB  0.836470  0.462916
# 59  CaliforniaHousing      MaxAbs+XGB  0.836470  0.462916
# 44  CaliforniaHousing      MinMax+XGB  0.836470  0.462916
# 54  CaliforniaHousing      Robust+XGB  0.836470  0.462916
# 62  CaliforniaHousing       MaxAbs+RF  0.802672  0.508508
# 47  CaliforniaHousing       MinMax+RF  0.802631  0.508561
# 52  CaliforniaHousing     Standard+RF  0.802624  0.508570
# 57  CaliforniaHousing       Robust+RF  0.802563  0.508649
# 53  CaliforniaHousing    Standard+SVR  0.752079  0.569980
# 58  CaliforniaHousing      Robust+SVR  0.694069  0.633163
# 65  CaliforniaHousing  OHE+RidgePoly2  0.665033  0.662528
# 48  CaliforniaHousing      MinMax+SVR  0.655280  0.672105
# (base) jaewoo000@JaeBook-Pro IBM:RedHat % 