import sklearn as sk
import tensorflow as tf
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from keras.optimizers import Adam, Adagrad, SGD
from keras.datasets import mnist

#1 data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

# x_train	Training images (60,000 samples)	(60000, 28, 28)
# y_train	Training labels (0~9)	(60000,)
# x_test	Test images (10,000 samples)	(10000, 28, 28)
# y_test	Test labels	(10000,)

x_train = x_train/255.
x_test = x_test/255.

# 2. 탐색할 옵티마이저 및 러닝레이트
optimizers = [Adam, Adagrad, SGD]
lr_list = [0.1, 0.01, 0.05, 0.001, 0.0005, 0.0001]

# 3. 결과 저장용 리스트
results = []

# 4. 탐색
for opt in optimizers:
    for lr in lr_list:
        model = Sequential([
            Dense(128, input_dim=8, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1)
        ])

        optimizer_instance = opt(learning_rate=lr)
        model.compile(loss='mse', optimizer=optimizer_instance)

        start = time.time()
        history = model.fit(
            x_train, y_train,
            epochs=10,
            batch_size=32,
            verbose=0,
            validation_split=0.2
        )
        end = time.time()

        final_val_loss = history.history['val_loss'][-1]
        results.append({
            'optimizer': opt.__name__,
            'lr': lr,
            'val_loss': final_val_loss,
            'time': end - start
        })

        print(f"[{opt.__name__} | lr={lr}] → val_loss: {final_val_loss:.6f} / time: {end - start:.2f} sec")

# 5. 최적 조합 출력
best = sorted(results, key=lambda x: x['val_loss'])[0]
print("\n✅ 가장 좋은 결과:")
print(f"Optimizer: {best['optimizer']}, Learning Rate: {best['lr']}")
print(f"Validation Loss: {best['val_loss']:.6f}, Time: {best['time']:.2f} sec")