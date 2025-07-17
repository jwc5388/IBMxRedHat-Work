
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import random
import tensorflow as tf
import datetime

# Seed 고정
seed = 33
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 경로 설정
if os.path.exists('/workspace/TensorJae/Study25/'):
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')

path = os.path.join(BASE_PATH, '_data/dacon/electricity/')
save_path = os.path.join(path, '_save/')

# 데이터 불러오기
buildinginfo = pd.read_csv(path + 'building_info.csv')
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
samplesub = pd.read_csv(path + 'sample_submission.csv')

# 결측치 처리: '-' → 0, float 변환
cols_to_clean = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for col in cols_to_clean:
    buildinginfo[col] = buildinginfo[col].replace('-', 0).astype(float)

# 날짜 파싱 및 파생변수
for df in [train, test]:
    df['일시'] = pd.to_datetime(df['일시'])
    df['hour'] = df['일시'].dt.hour
    df['dayofweek'] = df['일시'].dt.dayofweek

# 건물 정보 병합
train = train.merge(buildinginfo, on='건물번호', how='left')
test = test.merge(buildinginfo, on='건물번호', how='left')

# 범주형 처리
train['건물유형'] = train['건물유형'].astype('category').cat.codes
test['건물유형'] = test['건물유형'].astype('category').cat.codes

# 피처 설정
test_features = ['건물번호', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)',
                 'ESS저장용량(kWh)', 'PCS용량(kW)', '기온(°C)', '강수량(mm)', '풍속(m/s)',
                 '습도(%)', 'hour', 'dayofweek']
target = '전력소비량(kWh)'

# 학습/테스트 분할
x = train[test_features]
y = np.log1p(train[target])
x_test_final = test[test_features]

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=seed)

# 스케일링
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_final_scaled = scaler.transform(x_test_final)

# 모델 정의
xgb = XGBRegressor(n_estimators=600, learning_rate=0.1, max_depth=6, random_state=seed)
lgb = LGBMRegressor(n_estimators=600, learning_rate=0.1, max_depth=6, random_state=seed)
cat = CatBoostRegressor(n_estimators=600, learning_rate=0.1, max_depth=6, random_state=seed, verbose=0)

stack_model = StackingRegressor(
    estimators=[
        ('xgb', xgb),
        ('lgb', lgb),
        ('cat', cat),
    ],
    final_estimator=GradientBoostingRegressor(n_estimators=350, random_state=seed),
    n_jobs=-1
)

# 학습
stack_model.fit(x_train_scaled, y_train)

# 예측 (로그 복원)
y_pred = np.expm1(stack_model.predict(x_val_scaled))
y_true = np.expm1(y_val)

# SMAPE 계산
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

print(f"\n✅ 검증 SMAPE: {smape(y_true, y_pred):.4f}")

# 제출
# 예측 및 로그 복원
final_pred = np.expm1(stack_model.predict(x_test_final_scaled))
samplesub['answer'] = final_pred

# 오늘 날짜
today = datetime.datetime.now().strftime('%Y%m%d')

# 검증 SMAPE 점수 계산
val_smape = smape(y_true, y_pred)
score_str = f"{val_smape:.4f}".replace('.', '_')

# 파일명 생성
filename = f"submission_{today}_SMAPE_{score_str}.csv"
file_path = os.path.join(path, filename)

# 저장
samplesub.to_csv(file_path, index=False)
print(f"📁 {filename} 저장 완료!")
