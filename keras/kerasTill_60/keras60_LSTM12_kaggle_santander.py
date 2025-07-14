import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import time

# 💡 재현성을 위한 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

# 1. 데이터 로드
path = '/workspace/TensorJae/Study25/_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv')
test_csv = pd.read_csv(path + 'test.csv')
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 2. 타겟/피처 분리
y = train_csv['target']
x = train_csv.drop(['ID_code', 'target'], axis=1)
x_submit = test_csv.drop(['ID_code'], axis=1)

# 3. 스케일링
scaler = MinMaxScaler()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_submit = scaler.transform(x_submit)  # ✅ 예측용 데이터도 변환

# 4. Reshape for LSTM
x_train = x_train.reshape(-1, 200, 1)
x_test = x_test.reshape(-1, 200, 1)

# 5. 모델 정의
model = Sequential([
    LSTM(64, return_sequences=True, dropout=0.2, input_shape=(200, 1)),
    LSTM(64, dropout=0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.summary()

# 6. 컴파일
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# 7. 콜백 정의
es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

# 8. 학습
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

# 9. 평가
loss, acc, auc = model.evaluate(x_test, y_test, verbose=0)
print("\n📊 Evaluation Metrics:")
print(f"✅ Loss     : {loss:.4f}")
print(f"✅ Accuracy : {acc:.4f}")
print(f"✅ AUC      : {auc:.4f}")
print(f"⏱️ 걸린시간  : {end - start:.2f}초")

# 10. scikit-learn 기반 추가 평가
y_pred_prob = model.predict(x_test, verbose=0)
print("✅ ROC AUC Score (sklearn):", roc_auc_score(y_test, y_pred_prob))
