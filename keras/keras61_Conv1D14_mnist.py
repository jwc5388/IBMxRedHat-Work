import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam

# ✅ 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

# ✅ 분산 전략 선언 (멀티 GPU 학습)
strategy = tf.distribute.MirroredStrategy()
print("🧠 Number of GPUs:", strategy.num_replicas_in_sync)

# ✅ 데이터셋 로딩
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ✅ Conv1D 입력을 위한 Reshape: (batch, steps, features)
# 여기선 가로 28픽셀을 시간축(steps)으로 간주, 각 줄(세로)이 feature
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(-1, 28, 28)  # shape = (60000, 28, 28)
x_test = x_test.reshape(-1, 28, 28)    # shape = (10000, 28, 28)

# ✅ 모델 정의 (strategy.scope 내부에서)
with strategy.scope():
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28)),
        BatchNormalization(),
        Dropout(0.3),

        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')  # MNIST는 10개 클래스
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# ✅ 모델 훈련
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=512,
    validation_data=(x_test, y_test)
)
