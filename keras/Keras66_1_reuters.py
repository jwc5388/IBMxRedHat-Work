from keras.datasets import reuters
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization,Embedding,Conv1D,Flatten
import pandas as pd


(x_train,y_train), (x_test,y_test) = reuters.load_data(
    num_words= 1000, # 단어사전의 개수, 빈도수가 높은 단어 순으로 1000개 뽑는다.
    test_split=0.2,
    # maxlen=200, # 단어 길이가 200개까지 있는 문장
)

print(x_train)
print(x_train.shape, x_test.shape) #(8982,) (2246,)
print(np.unique(y_train))

print(type(x_train))
print(type(x_train[0]))

print('뉴스기사의 최대길이 :', max(len(i) for i in x_train))
print('뉴스기사의 최소길이 :', min(len(i) for i in x_train))
print('뉴스기사의 평균길이 :', sum(map(len, x_train))/len(x_train))


exit()
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
# softmax onehot!
# (8982,100) 으로
#### 패딩 ####
from keras.preprocessing.sequence import pad_sequences
padding_x = pad_sequences(x_train,
                          padding='pre',
                          maxlen= 80
                          )
padding_test_x = pad_sequences(x_test,
                               padding='pre',
                               maxlen= 80)
print(padding_x.shape)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
padding_y = y_train.reshape(-1,1)
padding_y = encoder.fit_transform(padding_y)

padding_y_test = y_test.reshape(-1,1)
padding_y_test = encoder.transform(padding_y_test)
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
# print(y_train.shape) # (8982,)
# exit()

#2. 모델 구성
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=200,input_length=80))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv1D(100, kernel_size=2,activation='relu'))
model.add(BatchNormalization())


model.add(Conv1D(50, kernel_size=2,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(20,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())


model.add(Dense(padding_y.shape[1], activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
from keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=60,verbose=1,
                   restore_best_weights= True)
hist = model.fit(padding_x,padding_y, epochs= 100, batch_size= 16,verbose=2,validation_split=0.1, callbacks=[es,])

#4. 평가 예측
loss = model.evaluate(padding_test_x,padding_y_test,verbose=1)
result = model.predict(padding_test_x) #원래의 y값과 예측된 y값의 비교
print(result.shape) #(2246, 46)
result = np.argmax(result, axis=1)
print('loss :',loss[0])
print('acc :',loss[1])
print('result:',result[:10])
# predict = model.predict(padding_pred) #원래의 y값과 예측된 y값의 비교
# # predict =  (predict > 0.5).astype(int)
# print(predict)

# loss : 1.887308120727539
# acc : 0.5427426695823669
# result: [ 3  1 19  4  4  3  4  3  3  3]
import tensorflow as tf
print(tf.__version__)