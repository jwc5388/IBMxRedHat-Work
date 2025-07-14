
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint

import time
import tensorflow as tf

# 1. Load Data
path = '/workspace/TensorJae/Study25/_data/kaggle/bank/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. Encode categorical features
le_geo = LabelEncoder()
le_gender = LabelEncoder()

train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])

test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

# 3. Drop unneeded columns
train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# 4. Separate features and target
x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

# (165034, 10)
# (165034,)

# exit()


# 6. Train-test split (after scaling)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=33
)
# 5. Apply MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

x_train = x_train.reshape(-1, 10, 1)
x_test = x_test.reshape(-1, 10, 1)



model = Sequential([
    Conv1D(filters = 64, kernel_size = 2, padding = 'same', input_shape = (10,1), activation = 'relu'),
    Conv1D(filters = 64, kernel_size = 2, activation= 'relu'),
    Conv1D(filters = 64, kernel_size = 2, activation= 'relu'),
    Flatten(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') 
])
model.summary()


model.compile(
    loss='binary_crossentropy',          # ✅ 이진 분류에 적합
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]  # ✅ 내부 AUC도 추가
)


es = EarlyStopping(
    monitor='val_loss',       # 기준: 검증 손실
    patience=10,              # 10 epoch 개선 없으면 멈춤
    mode='min',               # 손실이므로 'min'
    verbose=1,
    restore_best_weights=True
)


start = time.time()

# 2. model.fit()에 callbacks 인자로 추가
hist = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[es]  
)



end = time.time()

# === 모델 평가 ===
# === 모델 평가 ===
loss, acc, auc = model.evaluate(x_test, y_test, verbose=0)
y_pred_proba = model.predict(x_test)

# === 임계값 기준 분류 (0.5 이상이면 1로 처리) ===
y_pred = (y_pred_proba > 0.5).astype(int)

# === 평가 지표 계산 ===
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

# === 결과 출력 ===
print(f'✅ Binary Crossentropy Loss : {loss:.4f}')
print(f'✅ Accuracy                 : {acc:.4f}')
print(f'✅ F1 Score                 : {f1:.4f}')
print(f'✅ ROC AUC                  : {roc_auc:.4f}')
print(f'⏱️ Training Time            : {end - start:.2f} sec')

print("\n📊 Confusion Matrix:")
print(cm)


# ✅ Binary Crossentropy Loss : 0.3358
# ✅ Accuracy                 : 0.8580
# ✅ F1 Score                 : 0.6218
# ✅ ROC AUC                  : 0.8808
# ⏱️ Training Time            : 54.10 sec

# 📊 Confusion Matrix:
# [[24467  1485]
#  [ 3202  3853]]

