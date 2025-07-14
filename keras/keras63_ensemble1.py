from os import name
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score

#1 data
x1_dataset = np.array([range(100), range(301,401)]).T               #(100,2)
                    #삼성전자 종가, 하이닉스 종가. 
x2_dataset = np.array([range(101,201), range(411,511), range(150,250)]).T       #(100,3)
                    #원류, 환율, 금시세
y = np.array(range(2001,2101))      #(100,)         화성의 화씨 온도


x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_dataset, x2_dataset, y, train_size=0.7, random_state=33)


print(x1_train.shape)  # (70, 2)
print(x2_train.shape)  # (70, 3)
print(y_train.shape)   # (70,)


#2-1 model
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name = 'IBM1')(input1)
dense2 = Dense(20, activation='relu', name = 'IBM2')(dense1)
dense3 = Dense(30, activation='relu', name = 'IBM3')(dense2)
dense4 = Dense(40, activation='relu', name = 'IBM4')(dense3)
output1 = Dense(50, activation='relu', name = 'IBM5')(dense4)

# model1 = Model(inputs = input1, outputs = output1)
# model1.summary()


#2-2 model
input2 = Input(shape=(3,))
dense21 = Dense(100, activation='relu', name = 'ibm21')(input2)
dense22 = Dense(50, activation='relu', name = 'ibm22')(dense21)
output2 = Dense(30, activation='relu', name = 'ibm23')(dense22)

# model2 = Model(inputs = input2, outputs = output2)


#2-3 model combine
from keras.layers import concatenate, Concatenate

merge1 = concatenate([output1, output2], name = 'mg1')
merge2 = Dense(40, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs = [input1, input2], outputs = last_output)
model.summary()


#3 compile
model.compile(loss = 'mse', optimizer = 'adam')


model.fit([x1_train, x2_train], y_train, epochs = 800, validation_split=0.2, verbose = 1, batch_size = 32 )


# 모델 평가 및 예측
pred = model.predict([x1_test, x2_test])
loss = model.evaluate([x1_test, x2_test], y_test, verbose=0)

# 지표 계산
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print('✅ Loss:', loss)
print('✅ RMSE:', rmse)
print('✅ R2 Score:', r2)

# 추가 예측 (새로운 데이터)
x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516), range(249,255)]).T
new_pred = model.predict([x1_pred, x2_pred])
print('Prediction:', new_pred.ravel())




 
# ✅ Loss: 0.0787869468331337
# ✅ RMSE: 0.2806901383857454
# ✅ R2 Score: 0.9999092758672379
# 1/1 [==============================] - 0s 21ms/step
# Prediction: [2099.5408 2101.6206 2103.708  2105.8562 2108.0051 2110.2139]