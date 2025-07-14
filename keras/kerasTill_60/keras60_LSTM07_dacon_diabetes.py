import numpy as np
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터 로딩
path = 'Study25/_data/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. 결측치 처리
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())

print("✅ Data Shape:", x.shape, y.shape)  # (652, 8) (652,)

# 3. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=33
)

# 4. 스케일링 + reshape
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 8, 1)  # (N, 8, 1)
x_test = x_test.reshape(-1, 8, 1)
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)

# 5. 모델 구성
model = Sequential([
    LSTM(64, return_sequences=True, dropout=0.2, input_shape=(8, 1)),
    LSTM(64, dropout=0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1)
])
model.summary()

# 6. 컴파일 및 콜백
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

es = EarlyStopping(
    monitor='val_loss',
    patience=15,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

# 7. 학습
start = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es],
    verbose=1
)
end = time.time()

# 8. 평가
loss, mae = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 9. 출력
print("\n📊 Evaluation Metrics:")
print(f"✅ Loss (MSE): {loss:.4f}")
print(f"✅ MAE       : {mae:.4f}")
print(f"✅ RMSE      : {rmse:.4f}")
print(f"✅ R² Score  : {r2:.4f}")
print(f"⏱️ 걸린시간: {end - start:.2f}초")
