# 🟢 DACON 당뇨병 예측 대회: 이진 분류 문제 (Outcome = 0 or 1)

import numpy as np
import pandas as pd
import time
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv1D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

# 1. 데이터 로딩
path = '/workspace/TensorJae/Study25/_data/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. 피처/타겟 분리 및 전처리
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

# 0을 NaN으로 처리 후 평균값 대체
x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())

# 3. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=33)

# 4. 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 5. reshape for Conv1D
x_train = x_train.reshape(-1, 8, 1)
x_test = x_test.reshape(-1, 8, 1)

# 6. 모델 구성
model = Sequential([
    Conv1D(filters=64, kernel_size=2, padding='same', input_shape=(8, 1), activation='relu'),
    Conv1D(filters=64, kernel_size=2, activation='relu'),
    Conv1D(filters=64, kernel_size=2, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # ✅ 이진 분류 → sigmoid
])

model.summary()

# 7. 컴파일
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# 8. 콜백
es = EarlyStopping(
    monitor='val_loss',
    patience=15,
    mode='min',
    verbose=1,
    restore_best_weights=True
)

# 9. 학습
start = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[es]
)
end = time.time()

# 10. 평가 및 예측
loss, acc, auc = model.evaluate(x_test, y_test, verbose=0)
y_pred_prob = model.predict(x_test)
y_pred_class = (y_pred_prob >= 0.5).astype(int)

# 11. 추가 지표
roc_auc = roc_auc_score(y_test, y_pred_prob)

# 12. 결과 출력
print("\n📊 Final Evaluation Metrics:")
print(f"✅ Binary Crossentropy Loss : {loss:.4f}")
print(f"✅ Accuracy                 : {acc:.4f}")
print(f"✅ Keras AUC                : {auc:.4f}")
print(f"✅ Sklearn ROC AUC          : {roc_auc:.4f}")
print(f"⏱️ Training Time            : {end - start:.2f} seconds")

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred_class))



# 📊 Final Evaluation Metrics:
# ✅ Binary Crossentropy Loss : 0.5079
# ✅ Accuracy                 : 0.7252
# ✅ Keras AUC                : 0.7754
# ✅ Sklearn ROC AUC          : 0.7751
# ⏱️ Training Time            : 13.90 seconds

# 📄 Classification Report:
#               precision    recall  f1-score   support

#            0       0.75      0.90      0.82        92
#            1       0.57      0.31      0.40        39

#     accuracy                           0.73       131
#    macro avg       0.66      0.60      0.61       131
# weighted avg       0.70      0.73      0.70       131