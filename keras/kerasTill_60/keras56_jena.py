import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime

# --- 1. 경로 설정 ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = f'/workspace/TensorJae/Study25/_save/jena/{timestamp}/'
os.makedirs(SAVE_DIR, exist_ok=True)

DATA_PATH = '/workspace/TensorJae/Study25/_data/jena_climate_2009_2016.csv'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'best_model.h5')
SUBMISSION_PATH = os.path.join(SAVE_DIR, 'submission.csv')

# --- 2. 데이터 로딩 및 구간 정의 ---
TIMESTEPS = 144  # 1일치 데이터 (144 * 10분)
df = pd.read_csv(DATA_PATH, parse_dates=['Date Time'])

start_pred = pd.to_datetime("31.12.2016 00:10:00", dayfirst=True)
end_pred = pd.to_datetime("01.01.2017 00:00:00", dayfirst=True)

# ✅ [수정] 미래 데이터 누수 해결
# 예측 시작 시점 이전의 데이터만 훈련/검증 데이터로 사용합니다.
df_train_val = df[df['Date Time'] < start_pred].copy()
df_to_predict = df[(df['Date Time'] >= start_pred) & (df['Date Time'] <= end_pred)].copy()

# --- 3. Feature 설정 및 순환성 변환 ---
feature_columns = [col for col in df.columns if col not in ['Date Time', 'wd (deg)']]
target_column = 'wd (deg)'

def deg_to_sin_cos(deg):
    rad = np.deg2rad(deg)
    return np.sin(rad), np.cos(rad)

df_train_val['wd_sin'], df_train_val['wd_cos'] = deg_to_sin_cos(df_train_val[target_column])

# --- 4. 정규화 ---
scaler = StandardScaler().fit(df_train_val[feature_columns])
df_train_val_scaled = df_train_val.copy()
df_train_val_scaled[feature_columns] = scaler.transform(df_train_val[feature_columns])

# --- 5. 슬라이딩 윈도우 생성 ---
def split_xy(x, y1, y2, timesteps):
    x_seq, y1_seq, y2_seq = [], [], []
    for i in range(len(x) - timesteps):
        x_seq.append(x[i:i+timesteps])
        y1_seq.append(y1[i+timesteps])
        y2_seq.append(y2[i+timesteps])
    return np.array(x_seq), np.array(y1_seq), np.array(y2_seq)

x_data = df_train_val_scaled[feature_columns].values
y_sin_data = df_train_val['wd_sin'].values
y_cos_data = df_train_val['wd_cos'].values

# ✅ [수정] stride=6 으로 설정 (1시간 간격 샘플링)
x, y_sin, y_cos = split_xy(x_data, y_sin_data, y_cos_data, TIMESTEPS)
y = np.stack([y_sin, y_cos], axis=1) # y 데이터를 하나로 합침

# --- 6. 학습/검증 분리 (시간 순서) ---
split_idx = int(len(x) * 0.9) # 90% 훈련, 10% 검증
x_train, x_val = x[:split_idx], x[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# --- 7. 모델 정의 ---
model = Sequential([
    LSTM(64, input_shape=(x.shape[1], x.shape[2]), return_sequences=True),
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(2, activation='tanh')  # sin, cos 두 개 예측
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# --- 8. 학습 ---
# ✅ EarlyStopping 추가
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)

model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=50, batch_size=64, callbacks=[es, checkpoint], verbose=1)

# --- 9. 예측 데이터 준비 ---
# 예측을 위해, 마지막 훈련 데이터(TIMESTEPS 만큼)와 예측할 구간의 데이터를 합칩니다.
last_train_data = df_train_val.tail(TIMESTEPS)
concat_for_pred = pd.concat([last_train_data, df_to_predict], ignore_index=True)

# 정규화 적용
concat_for_pred_scaled = concat_for_pred.copy()
concat_for_pred_scaled[feature_columns] = scaler.transform(concat_for_pred[feature_columns])
x_pred_data = concat_for_pred_scaled[feature_columns].values

# 예측할 시퀀스 생성
x_pred = []
# 전체 길이에서 TIMESTEPS를 뺀 만큼만 반복하면 정확히 예측 대상 기간의 샘플이 생성됨
for i in range(len(x_pred_data) - TIMESTEPS):
    x_pred.append(x_pred_data[i:i+TIMESTEPS])
x_pred = np.array(x_pred)

# --- 10. 예측 수행 ---
best_model = load_model(MODEL_SAVE_PATH)
y_pred_sin_cos = best_model.predict(x_pred)

# --- 11. 예측 결과 복원 및 평가 ---
def sincos_to_deg(sin_vals, cos_vals):
    radians = np.arctan2(sin_vals, cos_vals)
    degrees = np.rad2deg(radians)
    return (degrees + 360) % 360  # 음수 각도 보정

y_pred_deg = sincos_to_deg(y_pred_sin_cos[:, 0], y_pred_sin_cos[:, 1])
y_true_deg = df_to_predict[target_column].values

rmse = np.sqrt(mean_squared_error(y_true_deg, y_pred_deg))
mae = mean_absolute_error(y_true_deg, y_pred_deg)

print("\n" + "="*50)
print("📌 최종 예측 결과 평가")
print(f"📈 RMSE: {rmse:.4f}")
print(f"📈 MAE : {mae:.4f}")
print("="*50 + "\n")

# --- 12. 저장 ---
submission_df = pd.DataFrame({
    'Date Time': df_to_predict['Date Time'],
    'wd (deg)': y_pred_deg
})
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"✅ submission 저장 완료: {SUBMISSION_PATH}")
print(submission_df.head())


# 📈 RMSE: 59.2611
# 📈 MAE : 41.7753


# 📈 RMSE: 68.4766
# 📈 MAE : 45.6391

# 📈 RMSE: 70.7903
# 📈 MAE : 42.0439

# 📈 RMSE: 59.7079
# 📈 MAE : 38.8619


# 📈 RMSE: 53.7864
# 📈 MAE : 35.0676

# 📌 최종 예측 결과 평가
# 📈 RMSE: 65.3606
# 📈 MAE : 42.8297
# ====================