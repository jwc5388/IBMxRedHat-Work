import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM
from keras.datasets import mnist
from keras.optimizers import Adam
import numpy as np

# ✅ 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

# ✅ 분산 전략 선언
strategy = tf.distribute.MirroredStrategy()
print("🧠 Number of GPUs:", strategy.num_replicas_in_sync)

# ✅ 데이터셋 로딩
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ✅ 데이터 정규화
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ✅ Reshape (🎯 여기가 중요!)
x_train = x_train.reshape(-1, 28*28, 1)
x_test = x_test.reshape(-1, 28*28, 1)

# ✅ 모델 정의
with strategy.scope():
    model = Sequential([
        LSTM(64, input_shape=(28*28, 1)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax'),
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])  # 🎯 acc 포함

# ✅ 학습
model.fit(x_train, y_train,
          epochs=3,
          batch_size=512,
          validation_data=(x_test, y_test))

# ✅ 평가
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"✅ Test Loss: {loss:.4f}")
print(f"✅ Test Accuracy: {acc:.4f}")
