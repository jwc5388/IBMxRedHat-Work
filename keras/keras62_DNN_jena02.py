"""
기존 데이터에서 y를 144(1일치) 를 맞춰야 하므로, 

Y 데이터를 위로 144개 shift 해서 
새로운 행렬데이터를 만들어서

144개를 예측한다.

"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# --- 하이퍼파라미터 ---
TIMESTEPS = 144

# --- 데이터 로딩 ---
df = pd.read_csv('/workspace/TensorJae/Study25/_data/jena_climate_2009_2016.csv', parse_dates=['Date Time'])

# --- 시간 파생 변수 ---
def add_time_features(df):
    df['hour'] = df['Date Time'].dt.hour
    df['dayofyear'] = df['Date Time'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    return df

df = add_time_features(df)

# --- X, Y 분리 ---
# X는 wd 제거
X_columns = [col for col in df.columns if col not in ['Date Time', 'wd (deg)', 'hour', 'dayofyear']]
X_raw = df[X_columns].values
Y_raw = df['wd (deg)'].values

# --- 마지막 하루 제거 ---
X_raw = X_raw[:-TIMESTEPS]              # 마지막 144개 row 제거
Y_shifted = Y_raw[TIMESTEPS:]           # wd를 위로 144개 shift

# --- 정규화 ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# --- train/val/test split ---
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_shifted, test_size=0.1, shuffle=False)

# --- 모델 정의 ---
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- 학습 ---
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, Y_train, validation_split=0.1, epochs=10, batch_size=64, callbacks=[es], verbose=1)

# --- 예측 및 평가 ---
Y_pred = model.predict(X_test).flatten()
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
mae = mean_absolute_error(Y_test, Y_pred)

print(f"✅ RMSE: {rmse:.4f} / MAE: {mae:.4f}")



# ✅ RMSE: 85.2390 / MAE: 68.8178
# ✅ RMSE: 86.2249 / MAE: 69.4522
# ✅ RMSE: 85.7745 / MAE: 69.9776