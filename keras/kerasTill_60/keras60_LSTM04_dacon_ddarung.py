import numpy as np
import pandas as pd
import time
import datetime

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터 로딩
path = '/workspace/TensorJae/Study25/_data/dacon/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

# 결측치 처리
train_csv = train_csv.dropna()
test_csv = test_csv.fillna(train_csv.mean())

# 피처, 타겟 분리
x = train_csv.drop(['count'], axis=1)   # (1459, 9)
y = train_csv['count']                  # (1459,)

# train-test 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=3333
)

# 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Reshape for LSTM
x_train = x_train.reshape(-1, 9, 1)      # (N, 9, 1)
x_test = x_test.reshape(-1, 9, 1)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

# 2. 모델 구성
model = Sequential([
    LSTM(64, return_sequences=True, dropout=0.2, input_shape=(9, 1)),
    LSTM(64, return_sequences=False, dropout=0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1)
])
model.summary()

# 3. 콜백 설정
date = datetime.datetime.now().strftime("%m%d_%H%M")
path_save = '/workspace/TensorJae/Study25/_modelsave/ddarung_lstm_'
filename = f"{path_save}{date}_{{epoch:04d}}-{{val_loss:.4f}}.h5"

es = EarlyStopping(monitor='val_loss', mode='min', patience=5,
                   restore_best_weights=True, verbose=1)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=filename,
    verbose=1
)

# 4. 컴파일 & 학습
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

start = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es, mcp],
    verbose=1
)
end = time.time()

# 5. 평가 및 지표 출력
loss, mae = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Evaluation Metrics:")
print(f"✅ Loss (MSE): {loss:.4f}")
print(f"✅ MAE: {mae:.4f}")
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ R² Score: {r2:.4f}")
print(f"⏱️ 걸린시간: {end - start:.2f}초")
