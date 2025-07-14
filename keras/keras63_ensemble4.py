from os import name
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data
x_dataset = np.array([range(100), range(301, 401)]).T  #(100, 2)
# 삼성전자 종가, 하이닉스 종가. 

y1 = np.array(range(2001, 2101))  #(100,) 화성의 화씨 온도
y2 = np.array(range(13001, 13101))  # 비트코인 가격

x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x_dataset, y1, y2, train_size=0.7, random_state=33)


# 2-1. Model 1
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='IBM1')(input1)
dense2 = Dense(20, activation='relu', name='IBM2')(dense1)
dense3 = Dense(30, activation='relu', name='IBM3')(dense2)
dense4 = Dense(40, activation='relu', name='IBM4')(dense3)
output1 = Dense(50, activation='relu', name='IBM5')(dense4)


# 2-4. Combine the models
from keras.layers import concatenate

merge1 = concatenate([output1], name='mg1')
merge2 = Dense(40, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
middle_output = Dense(1, name = 'lastm')(merge3)


lastoutput11 = Dense(10,name= 'last11')(middle_output)
lastoutput12 = Dense(10, name='last12')(lastoutput11)
lastoutput13 = Dense(1, name = 'last13')(lastoutput12)

lastoutput21 = Dense(1, name = 'last21')(middle_output)

model = Model(inputs=[input1], outputs=[lastoutput13, lastoutput21])



# 3. Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    restore_best_weights=True,
    patience = 20
)

# 4. Train the model
model.fit(x_train, [y1_train, y2_train], epochs=1000, validation_split=0.2, verbose=1, batch_size=32, callbacks = [es])

# 5. Evaluate and predict
# 예측
pred1, pred2 = model.predict([x_test])

# 평가
loss = model.evaluate([x_test], [y1_test, y2_test], verbose=1)

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
# x2_pred = np.array([range(200, 206), range(510, 516), range(249, 255)]).T
# x3_pred = np.array([range(100, 106), range(400, 406), range(177, 183), range(133, 139)]).T
new_pred1, new_pred2 = model.predict([x1_pred])
print('📈 New Prediction (Mars temp):', new_pred1.ravel())
print('📈 New Prediction (Bitcoin):   ', new_pred2.ravel())


# ✅ Total Loss (MSEs): [773.8040161132812, 749.0011596679688, 24.8028621673584, 23.327180862426758, 3.9716796875]
# ✅ RMSE1 (Mars temp): 27.3679, R2_1: 0.1375
# ✅ RMSE2 (Bitcoin):   4.9802, R2_2: 0.9714
# 1/1 [==============================] - 0s 23ms/step
# 📈 New Prediction (Mars temp): [2051.9377 2054.4617 2056.9885 2059.514  2062.0364 2064.5598]
# 📈 New Prediction (Bitcoin):    [13093.044 13109.154 13125.265 13141.375 13157.485 13173.596]

