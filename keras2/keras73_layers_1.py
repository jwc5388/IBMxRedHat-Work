import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Conv1D, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import numpy as np
import time


import random

SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from keras.applications import VGG16, VGG19, ResNet50, ResNet50V2, ResNet101
from keras.applications import ResNet152, ResNet152V2, DenseNet121, DenseNet169
from keras.applications import DenseNet201, InceptionV3, InceptionResNetV2
from keras.layers import Dense, Flatten, AveragePooling2D


# 1. Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


vgg16 = VGG16(
    include_top=False,
    input_shape=(32,32,3),

)

vgg16.trainable =True         #가중치 안동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

# model.compile(loss= 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True,verbose=1)
# lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

# start = time.time()
# model.fit(x_train, y_train,
#           epochs=100,
#           batch_size=32,
#           validation_split=0.2,
#           callbacks=[es,lr],
#           verbose=1)
# end = time.time()

# loss, acc = model.evaluate(x_test, y_test, verbose=0)
# print('evaluationnnnnnnnnnn')
# print(f'loss: {loss:.4f}')
# print(f'accuracy: {acc:.4f}')
# print(f'Time: {end - start:.2f}seconds')

# y_pred= model.predict(x_test)
# y_pred = np.argmax(y_pred, axis=1)
# y_test = np.argmax(y_test, axis=1)

# acc_score = accuracy_score(y_test, y_pred)
# print(f'accuracy score: {acc_score}:.4f')



# # 가중치 동결 했을때,

# ### 실습 ###
# #비교할거
# # 1. 이번의 본인이 한 최상의 결과가
# # 2. 가중치를 동결하지 않고 훈련시켰을때, trainable = True
# # 3. 가중치를 동결하고 훈련시켰을때, trainable = False
# # 시간까지 비교하기





###추가###
# Flatten 과 GAP



###
# cifar10
# cifar100
# horse

