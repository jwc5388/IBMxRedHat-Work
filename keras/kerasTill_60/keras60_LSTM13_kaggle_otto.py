import time
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# 1. Load Data
path = '/workspace/TensorJae/Study25/_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# 2. Feature & Target 분리
x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

# 3. Label 인코딩 후 One-hot
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_ohe = to_categorical(y_encoded, num_classes=9)

# 4. Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y_ohe, test_size=0.2, random_state=42
)

# 5. Standard Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_submit = scaler.transform(test_csv)

# 6. LSTM용 reshape
x_train = x_train.reshape(-1, 93, 1)
x_test = x_test.reshape(-1, 93, 1)
x_submit = x_submit.reshape(-1, 93, 1)

# 7. Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, dropout=0.2, input_shape=(93, 1)),
    LSTM(64, dropout=0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(9, activation='softmax')  # ✅ 9개 클래스
])
model.summary()

# 8. Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 9. EarlyStopping 설정
es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

# 10. Train
start = time.time()
model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    callbacks=[es],
    verbose=1
)
end = time.time()

# 11. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"\n✅ Loss     : {loss:.4f}")
print(f"✅ Accuracy : {acc:.4f}")
print(f"⏱️ 걸린시간  : {end - start:.2f}초")

# 12. 예측 + 제출
preds = model.predict(x_submit)
submission_df = pd.DataFrame(preds, columns=submission_csv.columns[1:])
submission_df.insert(0, 'id', submission_csv['id'])
submission_df.to_csv(path + 'otto_lstm_submission.csv', index=False)
print("📁 저장 완료: otto_lstm_submission.csv")
