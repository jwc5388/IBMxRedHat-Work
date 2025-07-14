# (15, 5)를 원핫하면? -> (15, 5 ,30)
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,LSTM
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pandas as pd
#1. 데이터
docs = [
    '너무 재미있다','참 최고에요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다.','한 번 더 보고 싶어요.','글쎄',
    '별로에요','생각보다 지루해요','연기가 어색해요',
    '재미없어요','너무 재미없다.','참 재밌네요.',
    '석준이 바보','준희 잘생겼다','이삭이 또 구라친다',
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

predict =['이삭이 참 잘생겼다']

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화예요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '
# 별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23, '석준이': 24, '바보': 25, '준희': 26, '잘생겼다': 27, '이삭이': 28, '또': 29, '구라친다': 30}

x = token.texts_to_sequences(docs)
x_text = token.texts_to_sequences(predict)

#[[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]

#### 패딩 ####
from keras.preprocessing.sequence import pad_sequences
padding_x = pad_sequences(x,
              padding='pre', # 'post' 디폴트는 프리
              maxlen=5, # 앞이 짤린다
              truncating='pre', # 'post' # 디폴트는 프리
              )
padding_pred = pad_sequences(x_text,
                             padding='pre', # 'post' 디폴트는 프리
                            maxlen=5, # 앞이 짤린다
                            truncating='pre', # 'post' # 디폴트는 프리
                             
                             )
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
# print(padding_x) #[ 0  2  3]
# print(padding_x.shape) #(15, 5)
padding_x = padding_x.reshape(-1,1)
padding_x = encoder.fit_transform(padding_x)
padding_x = padding_x[:, 1:]
padding_x = padding_x.reshape(15,5,30)

padding_pred = padding_pred.reshape(-1,1)
padding_pred = encoder.transform(padding_pred)
padding_pred = padding_pred[:, 1:]
padding_pred = padding_pred.reshape(1,5,30)

x_train, x_test, y_train, y_test = train_test_split(padding_x,labels,test_size=0.1,random_state=42)

#2. 모델 구성
model = Sequential()
model.add(LSTM(40, input_shape=(5,30), activation='relu'))
model.add(Dense(20,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
from keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=60,verbose=1,
                   restore_best_weights= True)
hist = model.fit(x_train,y_train, epochs= 100, batch_size= 3, callbacks=[es])

#4. 평가 예측
loss = model.evaluate(x_test,y_test,verbose=1)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
result =  (result > 0.5).astype(int)
print('loss :',loss[0])
print('acc :',loss[1])
print('result:',result)
predict = model.predict(padding_pred) #원래의 y값과 예측된 y값의 비교
predict =  (predict > 0.5).astype(int)
print(predict)
def call(predict):
    if predict == 0:
        print('부정')
    else:
        print('긍정')
call(predict)