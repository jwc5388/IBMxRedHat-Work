import time
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# =======================
# 1. Load & Prepare Data
# =======================
path = '/workspace/TensorJae/Study25/_data/kaggle/otto/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# Features & Labels 분리
x = train_csv.drop('target', axis=1)
y = train_csv['target']

# 라벨 인코딩 + 원핫 인코딩
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y, num_classes=9)  # (61878, 9)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

# =======================
# 2. Scaling & Reshaping
# =======================
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_submit = scaler.transform(test_csv)

# Conv1D 입력 형식으로 reshape → (batch, timesteps, features)
# 여기선 (93,) → (93, 1)
x_train = x_train.reshape(-1, 93, 1)
x_test = x_test.reshape(-1, 93, 1)
x_submit = x_submit.reshape(-1, 93, 1)

print("x_train shape:", x_train.shape)  # (49502, 93, 1)
print("y_train shape:", y_train.shape)  # (49502, 9)

# =======================
# 3. Conv1D Model
# =======================
model = Sequential()

# Conv1D Layer 1
model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same', input_shape=(93, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Conv1D Layer 2
model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Conv1D Layer 3
model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Flatten & Output
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(9, activation='softmax'))  # 다중분류

# =======================
# 4. Compile & Train
# =======================
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

start = time.time()
model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[es],
    verbose=1
)
end = time.time()

# =======================
# 5. Evaluate & Predict
# =======================
loss, acc = model.evaluate(x_test, y_test)
print(f'✅ loss: {loss:.4f}, accuracy: {acc:.4f}')
print('🕒 걸린 시간:', round(end - start, 2), '초')


# ✅ loss: 0.5887, accuracy: 0.7724
# 🕒 걸린 시간: 15.75 초

# =======================
# 6. Submission
# =======================
preds = model.predict(x_submit)  # shape: (144368, 9)
submission_df = pd.DataFrame(preds, columns=submission_csv.columns[1:])
submission_df.insert(0, 'id', submission_csv['id'])
submission_df.to_csv(path + 'otto_conv1d_submission.csv', index=False)
print("📄 제출 파일 저장 완료: otto_conv1d_submission.csv")
