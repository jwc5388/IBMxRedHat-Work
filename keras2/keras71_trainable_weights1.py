import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import random

SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(tf.__version__)

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

model = Sequential()
model.add(Dense(3,input_dim = 1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ dense (Dense)                        │ (None, 3)                   │               6 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 2)                   │               8 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_2 (Dense)                      │ (None, 1)                   │               3 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 17 (68.00 B)
#  Trainable params: 17 (68.00 B)
#  Non-trainable params: 0 (0.00 B)


print(model.weights)

# [<Variable path=sequential/dense/kernel, shape=(1, 3), dtype=float32, value=[[ 0.8516406  -0.52920127 -0.9112464 ]]>, <Variable path=sequential/dense/bias, shape=(3,), dtype=float32, value=[0. 0. 0.]>, <Variable path=sequential/dense_1/kernel, shape=(3, 2), dtype=float32, value=[[ 0.6380346  -1.0862008 ]
#  [-0.38601977 -0.21482044]
#  [ 0.20393836 -0.87937653]]>, <Variable path=sequential/dense_1/bias, shape=(2,), dtype=float32, value=[0. 0.]>, <Variable path=sequential/dense_2/kernel, shape=(2, 1), dtype=float32, value=[[-0.5044266 ]
#  [ 0.90831745]]>, <Variable path=sequential/dense_2/bias, shape=(1,), dtype=float32, value=[0.]>]


print('=======================================================')
print(model.trainable_weights)
print('=======================================================')

# [<Variable path=sequential/dense/kernel, shape=(1, 3), dtype=float32, value=[[ 0.8516406  -0.52920127 -0.9112464 ]]>, <Variable path=sequential/dense/bias, shape=(3,), dtype=float32, value=[0. 0. 0.]>, <Variable path=sequential/dense_1/kernel, shape=(3, 2), dtype=float32, value=[[ 0.6380346  -1.0862008 ]
#  [-0.38601977 -0.21482044]
#  [ 0.20393836 -0.87937653]]>, <Variable path=sequential/dense_1/bias, shape=(2,), dtype=float32, value=[0. 0.]>, <Variable path=sequential/dense_2/kernel, shape=(2, 1), dtype=float32, value=[[-0.5044266 ]
#  [ 0.90831745]]>, <Variable path=sequential/dense_2/bias, shape=(1,), dtype=float32, value=[0.]>]


#weight나 trainable weight 나 갯수는 같다

print(len(model.weights))           #6
print(len(model.trainable_weights)) #6

## huggingface 가중치 갖다 써 


############################# 동결 #############################
model.trainable = False             # 동결🌟 🌟 🌟 🌟 🌟  

############################# 동결 #############################
print(len(model.weights))               #6
print(len(model.trainable_weights))     #0

#역전파 x = 가중치 갱신을 하지 않겠다.


print('===================================')
print(model.weights)

print('===================================')
print(model.trainable_weights)              #[]
print('===================================')

# model.summary()