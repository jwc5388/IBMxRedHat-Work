from os import name
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data
x1_dataset = np.array([range(100), range(301, 401)]).T  #(100, 2)
# 삼성전자 종가, 하이닉스 종가. 
x2_dataset = np.array([range(101, 201), range(411, 511), range(150, 250)]).T  #(100, 3)
# 원류, 환율, 금시세
x3_dataset = np.array([range(100), range(301, 401), range(77, 177), range(33, 133)]).T  #(100, 4)

y1 = np.array(range(2001, 2101))  #(100,) 화성의 화씨 온도
y2 = np.array(range(13001, 13101))  # 비트코인 가격

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_dataset, x2_dataset, x3_dataset, y1, y2, train_size=0.7, random_state=33)

# 2-1. Model 1
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='IBM1')(input1)
dense2 = Dense(20, activation='relu', name='IBM2')(dense1)
dense3 = Dense(30, activation='relu', name='IBM3')(dense2)
dense4 = Dense(40, activation='relu', name='IBM4')(dense3)
output1 = Dense(50, activation='relu', name='IBM5')(dense4)

# 2-2. Model 2
input2 = Input(shape=(3,))
dense21 = Dense(100, activation='relu', name='ibm21')(input2)
dense22 = Dense(50, activation='relu', name='ibm22')(dense21)
output2 = Dense(30, activation='relu', name='ibm23')(dense22)

# 2-3. Model 3
input3 = Input(shape=(4,))
dense31 = Dense(100, activation='relu', name='ibm31')(input3)
dense32 = Dense(50, activation='relu', name='ibm32')(dense31)
dense33 = Dense(50, activation='relu', name='ibm33')(dense32)
output3 = Dense(20, activation='relu', name='ibm34')(dense33)

# 2-4. Combine the models
from keras.layers import concatenate

merge1 = concatenate([output1, output2, output3], name='mg1')
merge2 = Dense(40, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
last_output1 = Dense(1, name='last')(merge3)

merge4 = concatenate([output1, output2, output3], name='mg4')
merge5 = Dense(40, name='mg5')(merge4)
merge6 = Dense(20, name='mg6')(merge5)
last_output2 = Dense(1, name='last1')(merge6)

###이 방법도 있다 ###
# 위 merge 인걸로!!!
# or
# middle_output = Dense(1, name = 'lastm')(merge3)

# 분리!!!
# lastoutput11 = Dense(10,name= 'last11')(middle_output)
# lastoutput12 = Dense(10, name='last12')(lastoutput11)
# lastoutput13 = Dense(1, name = 'last13')(lastoutput12)

# 분리 2 -> y2
# lastoutput21 = Dense(1, name = 'last21')(middle_output)

# model = Model(inputs=[input1, input2, input3], outputs=[last_output13, last_output21])


model = Model(inputs=[input1, input2, input3], outputs=[last_output1, last_output2])
model.summary()

# 3. Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# 4. Train the model
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=1000, validation_split=0.2, verbose=1, batch_size=32)

# 5. Evaluate and predict
# 예측
pred1, pred2 = model.predict([x1_test, x2_test, x3_test])

# 평가
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test], verbose=1)

# pred1, pred2는 각각 shape (30, 1) → flatten
y_pred1 = pred1.ravel()
y_pred2 = pred2.ravel()

# 실제값도 flatten
y_true1 = y1_test.ravel()
y_true2 = y2_test.ravel()

# 개별 RMSE 및 R²
rmse1 = np.sqrt(mean_squared_error(y_true1, y_pred1))
rmse2 = np.sqrt(mean_squared_error(y_true2, y_pred2))

r2_1 = r2_score(y_true1, y_pred1)
r2_2 = r2_score(y_true2, y_pred2)

# 출력
print(f'✅ Total Loss (MSEs): {loss}')
print(f'✅ RMSE1 (Mars temp): {rmse1:.4f}, R2_1: {r2_1:.4f}')
print(f'✅ RMSE2 (Bitcoin):   {rmse2:.4f}, R2_2: {r2_2:.4f}')


# 7. Additional prediction (new data)
x1_pred = np.array([range(100, 106), range(400, 406)]).T
x2_pred = np.array([range(200, 206), range(510, 516), range(249, 255)]).T
x3_pred = np.array([range(100, 106), range(400, 406), range(177, 183), range(133, 139)]).T
new_pred1, new_pred2 = model.predict([x1_pred, x2_pred, x3_pred])
print('📈 New Prediction (Mars temp):', new_pred1.ravel())
print('📈 New Prediction (Bitcoin):   ', new_pred2.ravel())


#3loss, 2mae
# ✅ Total Loss (MSEs): [17.957019805908203, 0.30735650658607483, 17.6496639251709, 0.16061605513095856, 1.1126627922058105]
# ✅ RMSE1 (Mars temp): 0.5544, R2_1: 0.9996
# ✅ RMSE2 (Bitcoin):   4.2012, R2_2: 0.9797
# 1/1 [==============================] - 0s 26ms/step
# 📈 New Prediction (Mars temp): [2101.595  2105.5757 2109.5562 2113.5388 2117.5225 2121.506 ]
# 📈 New Prediction (Bitcoin):    [13112.373 13135.955 13159.541 13183.135 13206.732 13230.332]