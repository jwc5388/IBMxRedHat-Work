import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout, BatchNormalization

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) # 시계열이 될 수도 있고 아닐 수도 있다. -> 데이터에 종속되지 마라

# 시계열 데이터라면 time_step에 따라 잘라내야함
x = np.array([[1,2,3],  # 다음은 4야~
              [2,3,4],  # 다음은 5야~
              [3,4,5],  # 다음은 6이야~
              [4,5,6],  # 다음은 7이야~
              [5,6,7],  # 다음은 8이야~
              [6,7,8],  # 다음은 9야~
              [7,8,9],  # 다음은 10이야~ -> 8,9,10 다음은 뭐게~? -> 이런 게 RNN의 작동 목적
             ])
y = np.array([ 4, 5, 6, 7, 8, 9,10])    # -> datasets 을 x, y로 이렇게 잘라야 함
# !!!time_step!!!이 클수록 잘 맞을 수도 있다.
# time_step은 중복될 수 있다.

# print(x.shape, y.shape)     #(7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
# print(x.shape)     #(7, 3, 1)   -> (batch_size, time_steps, feature)

# x = np.array([[[1],[2],[3]],  # 다음은 4야~
#               [[2],[3],[4]],  # 다음은 5야~
#               [[3],[4],[5]],  # 다음은 6이야~
#               [[4],[5],[6]],  # 다음은 7이야~
#               [[5],[6],[7]],  # 다음은 8이야~
#               [[6],[7],[8]],  # 다음은 9야~
#               [[7],[8],[9]],  # 다음은 10이야~ -> 8,9,10 다음은 뭐게~? 
#              ])       -> (7, 3, 1)


# 2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3,1), activation='relu'))    # 시계열 데이터의 아웃풋은 2차원이다. 시계열 데이터가 섞이면서 순서가 사라짐
# model.add(LSTM(10, input_shape=(3,1),))
model.add(GRU(10, input_shape=(3,1), activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.summary()
# SimpleRNN summary
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 10)                120
#  dense (Dense)               (None, 5)                 55
#  dense_1 (Dense)             (None, 1)                 6
# =================================================================
# Total params: 181
# Trainable params: 181
# Non-trainable params: 0
# _________________________________________________________________

# params = units * feature + units * units + bias(1) * units
#        = units(feature + units + bias(1))

# LSTM
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm (LSTM)                 (None, 10)                480
#  dense (Dense)               (None, 5)                 55
#  dense_1 (Dense)             (None, 1)                 6
# =================================================================
# Total params: 541
# Trainable params: 541
# Non-trainable params: 0
# _________________________________________________________________

# GRU
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  gru (GRU)                   (None, 10)                390
#  dense (Dense)               (None, 5)                 55
#  dense_1 (Dense)             (None, 1)                 6
# =================================================================
# Total params: 451
# Trainable params: 451
# Non-trainable params: 0
# _________________________________________________________________
# PS C:\Study25> 









