import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import math
# --- 1. 경로 설정 ---
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

DATA_PATH = '/workspace/TensorJae/Study25/_data/jena_climate_2009_2016.csv'
MODEL_SAVE_PATH = f'/workspace/TensorJae/Study25/_save/jena/best_model_{timestamp}.h5'
SUBMISSION_PATH = f'/workspace/TensorJae/Study25/_save/jena/submission_{timestamp}.csv'
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# --- 2. 데이터 로딩 ---
df = pd.read_csv(DATA_PATH, parse_dates=['Date Time'])

# --- 3. 예측 대상 구간 정의 ---
start_pred = pd.to_datetime("31.12.2016 00:10:00", dayfirst=True)
end_pred = pd.to_datetime("01.01.2017 00:00:00", dayfirst=True)

df_to_predict = df[(df['Date Time'] >= start_pred) & (df['Date Time'] <= end_pred)].copy()
df_train_val = df[~((df['Date Time'] >= start_pred) & (df['Date Time'] <= end_pred))].copy()

# --- 4. Feature 설정 ---
exclude_cols = ['Date Time', 'wd (deg)']
feature_columns = [col for col in df.columns if col not in exclude_cols]
target_column = 'wd (deg)'
TIMESTEPS = 24

# --- 5. 타겟 순환성 변환 ---
def deg_to_sin_cos(deg):
    rad = np.deg2rad(deg)
    return np.sin(rad), np.cos(rad)

df_train_val['wd_sin'], df_train_val['wd_cos'] = deg_to_sin_cos(df_train_val[target_column])
df_to_predict['wd_sin'], df_to_predict['wd_cos'] = deg_to_sin_cos(df_to_predict[target_column])

# --- 6. 정규화 (모든 입력에 StandardScaler 적용) ---
scaler = StandardScaler().fit(df_train_val[feature_columns])
df_train_val_scaled = df_train_val.copy()
df_train_val_scaled[feature_columns] = scaler.transform(df_train_val[feature_columns])

# --- 7. 슬라이딩 윈도우 생성 ---
def split_xy(x, y1, y2, timesteps):
    x_seq, y1_seq, y2_seq = [], [], []
    for i in range(len(x) - timesteps):
        x_seq.append(x[i:i+timesteps])
        y1_seq.append(y1[i+timesteps])
        y2_seq.append(y2[i+timesteps])
    return np.array(x_seq), np.array(y1_seq), np.array(y2_seq)

x_data = df_train_val_scaled[feature_columns].values
y_sin = df_train_val['wd_sin'].values
y_cos = df_train_val['wd_cos'].values
x, y_sin, y_cos = split_xy(x_data, y_sin, y_cos, TIMESTEPS)

# --- 8. 학습/검증 분리 ---
split_idx = int(len(x) * 0.8)
x_train, x_val = x[:split_idx], x[split_idx:]
y_sin_train, y_sin_val = y_sin[:split_idx], y_sin[split_idx:]
y_cos_train, y_cos_val = y_cos[:split_idx], y_cos[split_idx:]

# --- 9. 모델 정의 ---
model = Sequential([
    LSTM(64, input_shape=(x.shape[1], x.shape[2]), return_sequences=False),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.2), 
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(2)  # sin, cos 두 개 예측
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- 10. 학습 ---
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)
model.fit(x_train, np.stack([y_sin_train, y_cos_train], axis=1),
          validation_data=(x_val, np.stack([y_sin_val, y_cos_val], axis=1)),
          epochs=20, batch_size=64, callbacks=[checkpoint], verbose=1)

# --- 11. 예측 데이터 준비 ---
last_24 = df_train_val.tail(TIMESTEPS)
concat_pred_input = pd.concat([last_24, df_to_predict], ignore_index=True)
concat_pred_scaled = concat_pred_input.copy()
concat_pred_scaled[feature_columns] = scaler.transform(concat_pred_scaled[feature_columns])
x_pred_seq = concat_pred_scaled[feature_columns].values

x_pred = []
for i in range(len(x_pred_seq) - TIMESTEPS):
    x_pred.append(x_pred_seq[i:i+TIMESTEPS])
x_pred = np.array(x_pred)

# --- 12. 예측 수행 ---
best_model = load_model(MODEL_SAVE_PATH)
y_pred_sin_cos = best_model.predict(x_pred)

# sin, cos → 각도로 복원
def sincos_to_deg(sin_vals, cos_vals):
    radians = np.arctan2(sin_vals, cos_vals)
    degrees = np.rad2deg(radians)
    degrees = (degrees + 360) % 360  # 음수 각도 보정
    return degrees

y_pred_deg = sincos_to_deg(y_pred_sin_cos[:, 0], y_pred_sin_cos[:, 1])
y_true_deg = df_to_predict['wd (deg)'].values

# --- 13. 평가 ---
rmse = np.sqrt(mean_squared_error(y_true_deg, y_pred_deg))
mae = mean_absolute_error(y_true_deg, y_pred_deg)

print("\n📌 순환성 적용된 모델 평가 결과:")
print(f"📈 RMSE: {rmse:.4f}")
print(f"📈 MAE : {mae:.4f}")

# --- 14. 저장 ---
submission_df = pd.DataFrame({
    'Date Time': df_to_predict['Date Time'],
    'wd (deg)': y_pred_deg
})
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"✅ submission 저장 완료: {SUBMISSION_PATH}")


# 📈 RMSE: 59.2611
# 📈 MAE : 41.7753