import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor

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

# ✅ 범주형 피처 정의
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

# ✅ 범주형 → 숫자형 인코딩
for col in cat_features:
    le = LabelEncoder()
    full_data = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(full_data)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# 학습/검증 데이터 분리
X = train.drop(columns=['ID', 'stress_score'])
y = train['stress_score']
X_test = test.drop(columns=['ID'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 개별 모델 정의
xgb_model = XGBRegressor(
    n_estimators=700, learning_rate=0.05, max_depth=6,
    objective='reg:absoluteerror', random_state=42
)

lgb_model = lgb.LGBMRegressor(
    n_estimators=700, learning_rate=0.05, max_depth=6,
    objective='mae', random_state=42
)

cat_model = CatBoostRegressor(
    iterations=700, learning_rate=0.04, depth=7,
    loss_function='MAE',
    cat_features=[X.columns.get_loc(col) for col in cat_features],
    verbose=0, random_seed=42
)

# 스태킹 앙상블 모델
stack_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ],
    final_estimator=Ridge(alpha=1.0),
    passthrough=True,  # 원본 피처도 메타 모델에 전달
    n_jobs=-1
)

# 모델 학습
stack_model.fit(X_train, y_train)

# 검증 예측
val_pred = stack_model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_pred)
print(f"✅ Stacking Validation MAE: {val_mae:.4f}")

# 테스트 예측 및 제출 파일 저장
test_pred = stack_model.predict(X_test)
submission['stress_score'] = test_pred

from datetime import datetime
now = datetime.now().strftime('%m%d_%H%M')
filename = f'submission_stack_{now}_{val_mae:.4f}.csv'
submission.to_csv(os.path.join(path, filename), index=False)
print(f"📁 Saved: {filename}")