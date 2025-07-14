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
path = '/workspace/TensorJae/Study25/_data/kaggle/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

# 2. 피처 및 타겟 정의
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)  # (10886, 8)
y = train_csv['count']                                        # (10886,)

# 3. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=333)

# 4. 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 5. Reshape for LSTM
x_train = x_train.reshape(-1, 8, 1)      # (N, 8, 1)
x_test = x_test.reshape(-1, 8, 1)
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)

# 6. 모델 구성 (LSTM only)
model = Sequential([
    LSTM(64, return_sequences=True, dropout=0.2, input_shape=(8, 1)),
    LSTM(64, dropout=0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1)  # ✅ count 예측은 단일 출력
])
model.summary()

# 7. 콜백 설정
date = datetime.datetime.now().strftime("%m%d_%H%M")
save_path = f'/workspace/TensorJae/Study25/_modelsave/bike_lstm_{date}_'
filename = save_path + '{epoch:04d}-{val_loss:.4f}.h5'

es = EarlyStopping(monitor='val_loss', mode='min', patience=5,
                   restore_best_weights=True, verbose=1)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=filename,
    verbose=1
)

# 8. 컴파일 및 학습
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

# 9. 평가
loss, mae = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 10. 결과 출력
print("\n📊 Evaluation Metrics:")
print(f"✅ Loss (MSE): {loss:.4f}")
print(f"✅ MAE       : {mae:.4f}")
print(f"✅ RMSE      : {rmse:.4f}")
print(f"✅ R² Score  : {r2:.4f}")
print(f"⏱️ 걸린시간: {end - start:.2f}초")
