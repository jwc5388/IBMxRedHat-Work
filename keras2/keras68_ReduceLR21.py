import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==============================
# 0) 경로
# ==============================
if os.path.exists('/workspace/TensorJae/Study25/'):
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')

DATA_PATH = os.path.join(BASE_PATH, '_data/jena/jena_climate_2009_2016.csv')
TIMESTEPS = 144      # 1일(10분 간격 x 144)
STRIDE = 1           # 필요시 6으로 바꾸면 1시간 간격 샘플링

# ==============================
# 1) 데이터 로딩 & 날짜 파싱 고정
# ==============================
df = pd.read_csv(DATA_PATH)
# 안전 파싱(형식 고정 + 실패 허용)
dt_raw = df['Date Time'].astype(str).str.strip()
dt = pd.to_datetime(dt_raw, format="%d.%m.%Y %H:%M:%S", errors='coerce', dayfirst=True)
if dt.isna().any():
    # 포맷이 섞였을 수 있으니 관대 파싱으로 한 번 더
    dt2 = pd.to_datetime(dt_raw[dt.isna()], errors='coerce', dayfirst=True)
    dt.loc[dt.isna()] = dt2
df['Date Time'] = dt.dt.tz_localize(None)

# 파싱 실패행 제거
bad = df['Date Time'].isna().sum()
if bad > 0:
    print(f"[경고] Date Time 파싱 실패 {bad}행 드랍")
    df = df[df['Date Time'].notna()].reset_index(drop=True)

start_pred = pd.to_datetime("31.12.2016 00:10:00", format="%d.%m.%Y %H:%M:%S", dayfirst=True)
end_pred   = pd.to_datetime("01.01.2017 00:00:00",   format="%d.%m.%Y %H:%M:%S", dayfirst=True)

# 누수 방지 분할
df_train_val  = df[df['Date Time'] < start_pred].copy()
df_to_predict = df[(df['Date Time'] >= start_pred) & (df['Date Time'] <= end_pred)].copy()

# ==============================
# 2) 피처/타깃 설정
# ==============================
target_col = 'wd (deg)'
if target_col not in df_train_val.columns:
    raise KeyError(f"타깃 컬럼 '{target_col}' 이(가) 데이터에 없습니다.")

# 수치형 피처만 사용(날짜/문자 자동 배제)
feature_columns = df_train_val.select_dtypes(include=[np.number]).columns.tolist()
if target_col in feature_columns:
    feature_columns.remove(target_col)

# 타깃 각도 → (sin, cos)
def deg_to_sin_cos(deg):
    rad = np.deg2rad(deg)
    return np.sin(rad), np.cos(rad)

df_train_val[target_col] = pd.to_numeric(df_train_val[target_col], errors='coerce')
na_tgt = df_train_val[target_col].isna().sum()
if na_tgt > 0:
    print(f"[경고] 타깃 NaN {na_tgt}행 드랍")
    df_train_val = df_train_val[df_train_val[target_col].notna()].reset_index(drop=True)

df_train_val['wd_sin'], df_train_val['wd_cos'] = deg_to_sin_cos(df_train_val[target_col])

# ==============================
# 3) 스케일링(수치형만)
# ==============================
scaler = StandardScaler()
# 스케일 전 결측/무한치 처리
X_num = df_train_val[feature_columns].replace([np.inf, -np.inf], np.nan)
num_na = X_num.isna().sum().sum()
if num_na > 0:
    print(f"[경고] 스케일 전 결측 {num_na}개 → 0 대체")
    X_num = X_num.fillna(0.0)

X_scaled = scaler.fit_transform(X_num)
df_train_val_scaled = df_train_val.copy()
df_train_val_scaled[feature_columns] = X_scaled

# ==============================
# 4) 시퀀스 생성
# ==============================
def make_sequences(x_arr, y1, y2, timesteps=144, stride=1):
    X, Y = [], []
    N = len(x_arr)
    for start in range(0, N - timesteps, stride):
        end = start + timesteps
        X.append(x_arr[start:end])
        Y.append([y1[end], y2[end]])  # 다음 시점 예측
    if len(X) == 0:
        raise ValueError("시퀀스가 생성되지 않았습니다. TIMESTEPS/STRIDE/데이터 길이를 확인하세요.")
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)

x_data = df_train_val_scaled[feature_columns].values
y_sin  = df_train_val['wd_sin'].values
y_cos  = df_train_val['wd_cos'].values

X, Y = make_sequences(x_data, y_sin, y_cos, timesteps=TIMESTEPS, stride=STRIDE)

# 시간순 분할(90/10)
split_idx = int(len(X) * 0.9)
x_train, x_val = X[:split_idx], X[split_idx:]
y_train, y_val = Y[:split_idx], Y[split_idx:]

print(f"[확인] X shape: {X.shape}, train: {x_train.shape}, val: {x_val.shape}")

# ==============================
# 5) 모델
# ==============================
model = Sequential([
    LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True),
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(2, activation='tanh')  # sin, cos
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

es  = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.9, verbose=1)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50, batch_size=64,
    callbacks=[es, rlr],
    verbose=1
)

# ==============================
# 6) 예측 시퀀스 준비(훈련 마지막 144 + 예측 구간)
# ==============================
last_train = df_train_val.tail(TIMESTEPS)
concat_for_pred = pd.concat([last_train, df_to_predict], ignore_index=True)

# 피처 스케일
X_pred_num = concat_for_pred[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_pred_scaled = scaler.transform(X_pred_num)
x_pred_data = X_pred_scaled

# 시퀀스 생성: 예측 구간 길이만큼 생성
def make_pred_sequences(x_arr, timesteps=144):
    Xp = []
    for i in range(len(x_arr) - timesteps):
        Xp.append(x_arr[i:i+timesteps])
    return np.asarray(Xp, dtype=np.float32)

x_pred = make_pred_sequences(x_pred_data, TIMESTEPS)

# ==============================
# 7) 예측 & 평가(원형 각도 지표 포함)
# ==============================
y_pred_sin_cos = model.predict(x_pred, verbose=0)

def sincos_to_deg(sin_vals, cos_vals):
    radians = np.arctan2(sin_vals, cos_vals)
    degrees = np.rad2deg(radians)
    return (degrees + 360) % 360

y_pred_deg = sincos_to_deg(y_pred_sin_cos[:, 0], y_pred_sin_cos[:, 1])

y_true_deg = df_to_predict[target_col].values.astype(float)
mask = ~np.isnan(y_true_deg)
y_true_deg = y_true_deg[mask]
y_pred_deg = y_pred_deg[:len(y_true_deg)]  # 방어적 슬라이싱

# 원형(360° 주기) 기반 오차
def circular_diff(a_deg, b_deg):
    d = np.abs(a_deg - b_deg) % 360
    return np.minimum(d, 360 - d)

c_rmse = np.sqrt(np.mean(circular_diff(y_true_deg, y_pred_deg) ** 2))
c_mae  = np.mean(circular_diff(y_true_deg, y_pred_deg))

# 일반 RMSE/MAE(참고)
rmse = np.sqrt(mean_squared_error(y_true_deg, y_pred_deg))
mae  = mean_absolute_error(y_true_deg, y_pred_deg)

print("\n" + "="*50)
print("📌 최종 예측 결과 평가")
print(f"📈 Circular RMSE: {c_rmse:.4f}")
print(f"📈 Circular MAE : {c_mae:.4f}")
print(f"(ref) RMSE      : {rmse:.4f}")
print(f"(ref) MAE       : {mae:.4f}")
print("="*50 + "\n")



# 📌 최종 예측 결과 평가
# 📈 Circular RMSE: 54.6024
# 📈 Circular MAE : 40.3133
# (ref) RMSE      : 61.7938
# (ref) MAE       : 42.6384