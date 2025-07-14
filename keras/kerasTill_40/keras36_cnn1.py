from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input, Conv2D


#원본은 (N,5,5,1) 이미지
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5, 5, 1))) # (N,4,4,10)

model.add(Conv2D(5, (2,2)))                         # (3,3,5)
# Conv2D(필터 수, 커널 크기, input_shape=(높이, 너비, 채널 수))
# 위에서 input_shape=(5,5,1)은 5x5 흑백 이미지(채널 1)를 의미

# model.add(Conv2D)

# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ conv2d (Conv2D)                      │ (None, 4, 4, 10)            │              50 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
model.summary()