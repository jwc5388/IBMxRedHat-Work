from keras.datasets import imdb
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Embedding, Conv1D, Dropout, Flatten


(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000,
    # test_split=0.2,
    
)

print(x_train)
print(y_train) #[1 0 0 ... 0 1 0]

print(x_train.shape, y_train.shape) #(25000,) (25000,)
print(np.unique(y_train, return_counts=True))   #(array([0, 1]), array([12500, 12500]))

# print(pd.value_counts(y_train))

# 1    12500
# 0    12500
# Name: count, dtype: int64

print('영화평의 최대길이 :', max(len(i) for i in x_train))  #2494
print('영화평의 최소길이 :', min(len(i) for i in x_train))  #11
print('영화평의 평균길이 :', sum(map(len, x_train))/len(x_train))   # 238.71364



from keras.preprocessing.sequence import pad_sequences
padding_x = pad_sequences(x_train,
                          padding='pre',
                          maxlen=240)
padding_test_x = pad_sequences(x_test, padding='pre',
                               maxlen=240)

print(padding_x.shape)  #(25000, 240)


# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder(sparse_output=False)
# padding_y = y_train.reshape(-1,1)
# padding_y = encoder.fit_transform(padding_y)

# padding_y_test= y_test.reshape(-1,1)
# padding_y_test = encoder.transform(padding_y_test)


#2. 모델 구성
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=200,input_length=240))
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


model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])

from keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=60,verbose=1,
                   restore_best_weights= True)
hist = model.fit(padding_x,y_train, epochs= 10, batch_size= 16,verbose=1,validation_split=0.1, callbacks=[es,])

#4. 평가 예측
loss = model.evaluate(padding_test_x,y_test,verbose=1)
result = model.predict(padding_test_x) 
print(result.shape) 
result = np.argmax(result, axis=1)
print('loss :',loss[0])
print('acc :',loss[1])
print('result:',result[:10])




# loss : 0.35175296664237976
# acc : 0.8593599796295166
# result: [0 0 0 0 0 0 0 0 0 0]
