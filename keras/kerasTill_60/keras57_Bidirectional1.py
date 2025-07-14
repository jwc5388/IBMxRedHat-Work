import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SimpleRNN, GRU, Bidirectional


#1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10]) # 시계열이 될 수도 있고 아닐 수도 있다. -> 데이터에 종속되지 마라

x = np.array([[1,2,3],  
              [2,3,4],  
              [3,4,5],  
              [4,5,6],  
              [5,6,7],  
              [6,7,8],  
              [7,8,9],  
             ])
y = np.array([ 4, 5, 6, 7, 8, 9,10])    # -> datasets 을 x, y로 이렇게 잘라야 함



model = Sequential()
model.add(Bidirectional(GRU(units = 10), input_shape = (3,1)))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()



#######param 개수########
"""
    RNN : 120
Bidirectional : 240


GRU 190
Bidirectional : 780 



LSTM 480
Bidirectional :

 Layer (type)                Output Shape              Param #   
=================================================================
 bidirectional (Bidirection  (None, 20)                960       
 al)                                                             
                                                                 
 dense (Dense)               (None, 7)                 147       
                                                                 
 dense_1 (Dense)             (None, 1)                 8         
                                                             

"""

