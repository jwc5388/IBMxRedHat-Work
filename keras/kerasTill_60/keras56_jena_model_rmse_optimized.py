import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import mean_squared_error

# --- 0. 함수 정의 ---
def deg_to_sin_cos(deg):
    rad = np.deg2rad(deg)
    return np.sin(rad), np.cos(rad)

def sincos_to_deg(sin_vals, cos_vals):
    radians = np.arctan2(sin_vals, cos_vals)
    degrees = np.rad2deg(radians)
    return (degrees + 360) % 360

def circular_rmse(y_true_deg, y_pred_deg):
    diff = np.abs(y_true_deg - y_pred_deg)
    diff = np.where(diff > 180, 360 - diff, diff)
    return np.sqrt(np.mean(diff ** 2))

def angular_loss(y_true, y_pred):
    y_true_angle = tf.atan2(y_true[:, 0], y_true[:, 1])
    y_pred_angle = tf.atan2(y_pred[:, 0], y_pred[:, 1])
    diff = y_true_angle - y_pred_angle
    return K.mean(1 - K.cos(diff))

# --- 1. 경로 설정 ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = f'/workspace/TensorJae/Study25/_save/jena/{timestamp}/'
os.makedirs(SAVE_DIR, exist_ok=True)

DATA_PATH = '/workspace/TensorJae/Study25/_data/jena_climate_2009_2016.csv'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'best_model.h5')
SUBMISSION_PATH = os.path.join(SAVE_DIR, 'submission.csv')

# --- 2. 데이터 로딩 및 구간 정의 ---
TIMESTEPS = 144
df = pd.read_csv(DATA_PATH, parse_dates=['Date Time'])

start_pred = pd.to_datetime("31.12.2016 00:10:00", dayfirst=True)
end_pred = pd.to_datetime("01.01.2017 00:00:00", dayfirst=True)

df_train_val = df[df['Date Time'] < start_pred].copy()
df_to_predict = df[(df['Date Time'] >= start_pred) & (df['Date Time'] <= end_pred)].copy()

# --- 3. Feature 설정 및 시간 파생 변수 추가 ---
target_column = 'wd (deg)'

def add_time_features(df):
    df['hour'] = df['Date Time'].dt.hour
    df['dayofyear'] = df['Date Time'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    return df

df_train_val = add_time_features(df_train_val)
df_to_predict = add_time_features(df_to_predict)

feature_columns = [col for col in df_train_val.columns if col not in ['Date Time', 'wd (deg)', 'hour', 'dayofyear']]

# --- 4. 풍향 변환 및 정규화 ---
df_train_val['wd_sin'], df_train_val['wd_cos'] = deg_to_sin_cos(df_train_val[target_column])

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

x, y_sin, y_cos = split_xy(x_data, y_sin_data, y_cos_data, TIMESTEPS)
y = np.stack([y_sin, y_cos], axis=1)

# --- 6. 학습/검증 분리 ---
split_idx = int(len(x) * 0.9)
x_train, x_val = x[:split_idx], x[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# --- 7. 모델 정의 ---
model = Sequential([
    LSTM(64, return_sequences=True, dropout=0.2, input_shape=(x.shape[1], x.shape[2])),
    LSTM(64, return_sequences=False, dropout=0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(2, activation='tanh')
])
model.compile(optimizer='adam', loss=angular_loss, metrics=['mae'])
model.summary()

# --- 8. 학습 ---
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)

model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=30, batch_size=64, callbacks=[es, checkpoint], verbose=1)

# --- 9. 예측 데이터 준비 ---
last_train_data = df_train_val.tail(TIMESTEPS)
concat_for_pred = pd.concat([last_train_data, df_to_predict], ignore_index=True)
concat_for_pred = add_time_features(concat_for_pred)
concat_for_pred_scaled = concat_for_pred.copy()
concat_for_pred_scaled[feature_columns] = scaler.transform(concat_for_pred[feature_columns])
x_pred_data = concat_for_pred_scaled[feature_columns].values

x_pred = []
for i in range(len(x_pred_data) - TIMESTEPS):
    x_pred.append(x_pred_data[i:i+TIMESTEPS])
x_pred = np.array(x_pred)

# --- 10. 예측 수행 ---
best_model = load_model(MODEL_SAVE_PATH, custom_objects={'angular_loss': angular_loss})
y_pred_sin_cos = best_model.predict(x_pred)

# --- 11. 결과 복원 및 평가 ---
y_pred_deg = sincos_to_deg(y_pred_sin_cos[:, 0], y_pred_sin_cos[:, 1])
y_true_deg = df_to_predict[target_column].values

# ✅ 일반 RMSE로 평가 (사용자 요구)
rmse = np.sqrt(mean_squared_error(y_true_deg, y_pred_deg))
mae = mean_absolute_error(y_true_deg, y_pred_deg)

print("\n" + "="*50)
print("📌 최종 예측 결과 평가")
print(f"📈 RMSE : {rmse:.4f}")
print(f"📈 MAE : {mae:.4f}")
print("="*50 + "\n")

# --- 12. 결과 저장 ---
submission_df = pd.DataFrame({
    'Date Time': df_to_predict['Date Time'],
    'wd (deg)': y_pred_deg
})
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"✅ submission 저장 완료: {SUBMISSION_PATH}")
print(submission_df.head())



# /workspace/TensorJae/Study25/_save/jena/20250620_020302/submission.csv
# 📌 최종 예측 결과 평가
# 📈 RMSE (circular): 54.7664
# 📈 MAE : 37.1576
# =======================


# 📌 최종 예측 결과 평가
# 📈 RMSE (circular): 54.4123
# 📈 MAE : 36.9871
# ✅ submission 저장 완료: /workspace/TensorJae/Study25/_save/jena/20250620_021212/submission.csv


# 📈 RMSE (circular): 52.3034
# 📈 MAE : 33.8783
# ==================================================

# ✅ submission 저장 완료: /workspace/TensorJae/Study25/_save/jena/20250620_021846/submission.csv


# 📌 최종 예측 결과 평가
# 📈 RMSE : 60.2832
# 📈 MAE : 40.0321
# ==================================================

# ✅ submission 저장 완료: /workspace/TensorJae/Study25/_save/jena/20250620_023540/submission.csv


# 📌 최종 예측 결과 평가
# 📈 RMSE : 54.1183
# 📈 MAE : 34.1738
# ==================================================

# ✅ submission 저장 완료: /workspace/TensorJae/Study25/_save/jena/20250620_024454/submission.csv



# 📌 최종 예측 결과 평가
# 📈 RMSE : 54.3033
# 📈 MAE : 33.0132
# ==================================================

# ✅ submission 저장 완료: /workspace/TensorJae/Study25/_save/jena/20250620_030203/submission.csv


# 📈 RMSE : 54.1594
# 📈 MAE : 34.0927
# ==================================================

# ✅ submission 저장 완료: /workspace/TensorJae/Study25/_save/jena/20250620_030923/submission.csv



# 📈 RMSE : 54.8842
# 📈 MAE : 35.4925
# ==================================================

# ✅ submission 저장 완료: /workspace/TensorJae/Study25/_save/jena/20250620_034508/submission.csv