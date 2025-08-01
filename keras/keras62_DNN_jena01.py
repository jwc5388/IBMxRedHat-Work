"""
jena의 rnn의 shape를 가져와서 단순 reshape 한다
shape = (42만, 타임스텝, 피쳐)
reshape = (42만, 타임스텝*피쳐)


"""


import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import mean_squared_error
import random
import torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


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

# 기존 RNN용 shape: (samples, 144, features)
n_samples = x.shape[0]
n_timesteps = x.shape[1]
n_features = x.shape[2]

# ✅ (수정1) RNN용 → DNN용으로 reshape
x = x.reshape(n_samples, n_timesteps * n_features)  # e.g. (420000, 1728)

split_idx = int(len(x) * 0.9)
x_train, x_val = x[:split_idx], x[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]




# --- 7. 모델 정의 ---
# ✅ (수정2) RNN 제거, Dense 기반 DNN 모델 정의
model = Sequential([
    Dense(256, input_dim=n_timesteps * n_features, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dense(2, activation='tanh')  # sin, cos 예측
])
model.compile(optimizer='adam', loss=angular_loss, metrics=['mae'])
model.summary()
# --- 8. 학습 ---
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)

model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=2, batch_size=64, callbacks=[es, checkpoint], verbose=1)

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

# ✅ 여기! DNN용으로 reshape
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1] * x_pred.shape[2])


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



# 📌 최종 예측 결과 평가
# 📈 RMSE : 56.4522
# 📈 MAE : 38.7400