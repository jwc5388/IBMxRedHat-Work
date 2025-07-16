import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from xgboost.callback import EarlyStopping
from xgboost.sklearn import XGBRegressor
# from xgboost import XGBRegressor
import os

# 경로 설정
if os.path.exists('/workspace/TensorJae/Study25/'):
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)
path = basepath + '_data/dacon/stress/'

# 데이터 로드
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')
test_ids = test['ID']

# ✅ 피처 엔지니어링
def preprocess(df):
    df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['bp_diff'] = df['systolic_blood_pressure'] - df['diastolic_blood_pressure']
    df['old'] = (df['age'] > 60).astype(int)
    df['overwork'] = (df['mean_working'] > 52).astype(int)
    df['metabolic'] = df['glucose'] + df['cholesterol'] + df['BMI']
    return df

train = preprocess(train)
test = preprocess(test)

# ✅ 범주형 피처
cat_features = [
    'gender', 'activity', 'smoke_status', 'medical_history',
    'family_medical_history', 'sleep_pattern', 'edu_level'
]

# 결측치 처리
for col in train.columns:
    if train[col].isnull().sum() > 0:
        fill = train[col].mode()[0] if col in cat_features else train[col].mean()
        train[col] = train[col].fillna(fill)
        test[col] = test[col].fillna(fill)

# ✅ 범주형 → 숫자형 인코딩 (LightGBM/XGBoost용)
for col in cat_features:
    le = LabelEncoder()
    full_data = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(full_data)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# 학습 데이터 준비
X = train.drop(columns=['ID', 'stress_score'])
y = train['stress_score']
X_test = test.drop(columns=['ID'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ CatBoost
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.04,
    depth=7,
    loss_function='MAE',
    cat_features=[X.columns.get_loc(col) for col in cat_features],
    verbose=0,
    random_seed=42
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

# ✅ LightGBM
lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    objective='mae',
    random_state=42
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=0)

# ✅ XGBoost
xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    objective='reg:absoluteerror',
    early_stopping_rounds=50,
    random_state=42
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=0
)

# ✅ 앙상블 예측 (검증)
val_pred = (
    cat_model.predict(X_val) +
    lgb_model.predict(X_val) +
    xgb_model.predict(X_val)
) / 3

val_mae = mean_absolute_error(y_val, val_pred)
print(f"✅ Ensemble Validation MAE: {val_mae:.4f}")

# ✅ 앙상블 예측 (테스트)
test_pred = (
    cat_model.predict(X_test) +
    lgb_model.predict(X_test) +
    xgb_model.predict(X_test)
) / 3

submission['stress_score'] = test_pred
submission.to_csv(path + 'submission_ensemble.csv', index=False)
print("📁 Saved: submission_ensemble.csv")