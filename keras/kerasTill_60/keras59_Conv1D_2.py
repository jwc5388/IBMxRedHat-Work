# from cuda.benchmarks.kernels import kernel_string
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, SimpleRNN, LSTM, GRU, Conv1D, Flatten
# from keras.callbacks import ModelCheckpoint

# #1. 데이터
# x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
#              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
#              [9,10,11],[10,11,12],[20,30,40],
#              [30,40,50],[40,50,60]])
# y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
# x_pred = np.array([50,60,70])

# print(x.shape, y.shape)     #(13, 3) (13,)

# x = x.reshape(x.shape[0],x.shape[1],1)
# x_pred = x_pred.reshape(1,3,1)

# model = Sequential()
# model.add(Conv1D(filters = 12, kernel_size = 2, padding = 'same', input_shape=(3,1)))
# model.add(Conv1D(filters = 11, kernel_size = 2))
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Dense(1))

# # model.summary()

# # exit()

# model.compile(loss='mse', optimizer='adam', metrics='acc')

# # path = './_save/keras53/mcp'
# model.fit(x, y, epochs=200)

# loss = model.evaluate(x, y)
# results = model.predict(x_pred)

# print('results :', results)
# print('Loss:', loss[0])
# print('Acc :', loss[1])






# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
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

# ✅ 데이터 정규화 (0~1 사이로 변환)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ✅ 모델 정의 (strategy.scope 내부에서)
with strategy.scope():
    model = Sequential([
        Conv1D(filters = 64, input_shape = (28*28,1), kernel_size = 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax'),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# ✅ 학습
model.fit(x_train, y_train,
          epochs=5,
          batch_size=512,
          validation_data=(x_test, y_test))
