from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from keras.callbacks import EarlyStopping
import numpy as np
import time

# 1. 데이터 로딩
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  # (442, 10), (442,)

# 2. 데이터 분할 및 reshape
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=333)
x_train = x_train.reshape(-1, 10, 1)
x_test = x_test.reshape(-1, 10, 1)

# 3. 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, padding='same', input_shape=(10, 1), activation='relu'))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  # regression: no activation

# 4. 컴파일
model.compile(loss='mse', optimizer='adam', metrics = ['mae'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    restore_best_weights=True,
    patience = 5
)

# 5. 학습
start = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=1, callbacks =[es])
end = time.time()

# 6. 평가 및 예측
loss,mae = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test)

# 7. 메트릭 계산
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 8. 출력
print("\n📊 Evaluation Metrics:")
print(f"✅ Loss (MSE): {loss:.4f}")
print(f"✅ RMSE      : {rmse:.4f}")
print(f"✅ MAE       : {mae:.4f}")
print(f"✅ R² Score  : {r2:.4f}")
print(f"⏱️ Training Time: {end - start:.2f}초")


# 📊 Evaluation Metrics:
# ✅ Loss (MSE): 2866.8635
# ✅ RMSE      : 53.5431
# ✅ MAE       : 44.1733
# ✅ R² Score  : 0.4880
# ⏱️ Training Time: 11.60초
