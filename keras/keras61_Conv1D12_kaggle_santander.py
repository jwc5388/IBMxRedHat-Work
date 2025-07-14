import numpy as np
import pandas as pd
import tensorflow as tf
import time
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Flatten, Conv1D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

# 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

# 데이터 로드
path = '/workspace/TensorJae/Study25/_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv')
test_csv = pd.read_csv(path + 'test.csv')
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 피처/타겟 분리
y = train_csv['target']
x = train_csv.drop(['ID_code', 'target'], axis=1)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Reshape for Conv1D
x_train = x_train.reshape(-1, 200, 1)
x_test = x_test.reshape(-1, 200, 1)

# 모델 구성
model = Sequential([
    Conv1D(64, kernel_size=2, padding='same', activation='relu', input_shape=(200,1)),
    Conv1D(64, kernel_size=2, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # ✅ 이진 분류
])

# 컴파일
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# EarlyStopping 콜백
es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

# 학습
start = time.time()
model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es],
    verbose=1
)
end = time.time()

# 평가
loss, acc, auc = model.evaluate(x_test, y_test, verbose=0)
y_pred_prob = model.predict(x_test, verbose=0)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# 결과 출력
print("\n📊 Final Evaluation Metrics:")
print(f"✅ Binary Crossentropy Loss : {loss:.4f}")
print(f"✅ Accuracy                : {acc:.4f}")
print(f"✅ Keras AUC               : {auc:.4f}")
print(f"✅ Sklearn ROC AUC         : {roc_auc:.4f}")
print(f"⏱️  Training Time           : {end - start:.2f}초")


# 📊 Final Evaluation Metrics:
# ✅ Binary Crossentropy Loss : 0.6686
# ✅ Accuracy                : 0.6417
# ✅ Keras AUC               : 0.8837
# ✅ Sklearn ROC AUC         : 0.8838
# ⏱️  Training Time           : 95.09초
