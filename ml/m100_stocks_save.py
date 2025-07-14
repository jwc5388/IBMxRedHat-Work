import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from joblib import dump


# 경로 설정
if os.path.exists('/workspace/TensorJae/Study25/'):
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')

path = os.path.join(BASE_PATH, '_data/stocks/')
samsung = pd.read_csv(path + 'samsung.csv', thousands=',')
hanwha = pd.read_csv(path + 'hanwha.csv', thousands=',')
hanwha = hanwha.rename(columns={'Unnamed: 6': '대비_금액'})

# 액면분할 보정 함수
def cut(df):
    df = df.copy()
    df['일자'] = pd.to_datetime(df['일자'], dayfirst=True)
    df = df.sort_values(by='일자', ascending=False).dropna()
    split_date = pd.to_datetime('2018-05-04')
    split_ratio = 50
    price_cols = ['시가', '고가', '저가', '종가']
    df.loc[df['일자'] < split_date, price_cols] = df.loc[df['일자'] < split_date, price_cols] / split_ratio
    return df

samsung = cut(samsung)
hanwha = cut(hanwha)

# 데이터 전처리 함수
def preprocess(df):
    df = df.copy()
    df['일자'] = pd.to_datetime(df['일자'], dayfirst=True)
    df = df.sort_values(by='일자', ascending=False).dropna()
    df['시가_5일_MA'] = df['시가'].rolling(window=5).mean()
    df['시가_변화'] = df['시가'].diff()
    df['등락률_1일전'] = df['시가'].pct_change() * 100
    df['target'] = df['시가'].shift(-1)
    # print(df['일자'].head(10))
    df = df.dropna()
    X = df.drop(columns=['일자', 'target', '대비'])
    y = df['target']
    latest_x = X.iloc[0]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    latest_x_scaled = scaler.transform([latest_x.values])[0]
    return X_scaled, y, latest_x_scaled, scaler

# 전처리
xh, yh, xh_latest, scaler_h = preprocess(hanwha)
xs, ys, xs_latest, scaler_s = preprocess(samsung)

# 훈련/테스트 분할
xh_train, xh_test, yh_train, yh_test = train_test_split(xh, yh, train_size=0.8, shuffle=False)
xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, train_size=0.8, shuffle=False)

# 한화 모델 학습
model_h = XGBRegressor(n_estimators=300, random_state=42)
model_h.fit(xh_train, yh_train)
pred_h_train = model_h.predict(xh_train)

# 삼성으로 잔차 예측
residuals = yh_train - pred_h_train
model_residual = XGBRegressor(n_estimators=300, random_state=42)
model_residual.fit(xs_train[:len(residuals)], residuals)

#============================================================================================
yh_pred_test = model_h.predict(xh_test)
rmse = np.sqrt(mean_squared_error(yh_test, yh_pred_test))
mae = mean_absolute_error(yh_test, yh_pred_test)
print("\n📊 한화 단독 모델 성능 ")
print(f" - RMSE: {rmse:,.2f}")
print(f" - MAE : {mae:,.2f}")

residual_test = model_residual.predict(xs_test[:len(yh_test)])
ensemble_pred_test = yh_pred_test + residual_test
rmse_ensemble = np.sqrt(mean_squared_error(yh_test, ensemble_pred_test))
mae_ensemble = mean_absolute_error(yh_test, ensemble_pred_test)
print("\n📊 ✅ 잔차 기반 앙상블 성능")
print(f" - RMSE: {rmse_ensemble:,.2f}")
print(f" - MAE : {mae_ensemble:,.2f}")

#============================================================================================

# 모델 및 스케일러 가중치 저장
save_path = os.path.join(path, '_save/')
os.makedirs(save_path, exist_ok=True)
dump(model_h, os.path.join(save_path, 'model_h.joblib'))
dump(model_residual, os.path.join(save_path, 'model_residual.joblib'))
dump(scaler_h, os.path.join(save_path, 'scaler_h.joblib'))
dump(scaler_s, os.path.join(save_path, 'scaler_s.joblib'))
print(f"\n✅ 모델 및 스케일러 가중치 저장 완료! → {save_path}")

#최종 예측
pred_h_latest = model_h.predict([xh_latest])[0]
residual_latest = model_residual.predict([xs_latest])[0]
final_pred = pred_h_latest + residual_latest
print("\n📈 2025년 7월 14일 시가 예측")
print(f" - 한화 단독 예측     : {int(pred_h_latest):,} 원")
print(f" - 삼성 기반 잔차 보정: {int(residual_latest):,} 원")
print(f" ✅ 최종 보정 예측     : {int(final_pred):,} 원")


# ==============================================================================
# [추가] 예측 결과 시각화 파트
# ==============================================================================
# plt.figure(figsize=(15, 7))

# # yh_test.index를 사용해 시간의 흐름에 따른 실제 주가와 예측 주가를 그립니다.
# plt.plot(yh_test.index, yh_test, label='실제 주가 (Actual Price)', color='blue', linewidth=2)
# plt.plot(yh_test.index, ensemble_pred_test, label='최종 예측 (Final Prediction)', color='red', linestyle='--', linewidth=2)
# # plt.plot(yh_test.index, yh_pred_test, label='기본 예측 (Base Prediction)', color='green', linestyle=':', linewidth=1) # 기본 예측과 비교하고 싶을 때 주석 해제

# plt.title('actual price vs. predicted price', fontsize=16)
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Price', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.show()
# ==============================================================================