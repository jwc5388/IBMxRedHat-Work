# Keras65_Embedding01_DNN1

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization ,Embedding,LSTM
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')
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
# print(token.word_index)
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

x_train, x_test, y_train, y_test = train_test_split(padding_x,labels,test_size=0.1,random_state=42)

#2. 모델 구성
model = Sequential()

# 임베딩 1.
# model.add(Embedding(input_dim=31, output_dim=100, input_length=5))
    # input_dim : 단어 사전의 개수(말뭉치의 개수)
    # output_dim : 다음 레이어로 전달하는 노드의 갯수 (조절가능)
    # input_length : (N,5), padding에 의해서 늘어난 칼럼의 수, 문장의 시퀀스 갯수

# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3100

#  lstm (LSTM)                 (None, 16)                7488

#  dense (Dense)               (None, 1)                 17

# =================================================================
# Total params: 10,605
# Trainable params: 10,605
# Non-trainable params: 0

# 임베딩 2.
# model.add(Embedding(input_dim=31, output_dim=100))
# input_length를 명시 안해줘도 알아서 조절해줌.
# =================================================================
#  embedding (Embedding)       (None, None, 100)         3100 => None 알아서 5로 맞춰준다

#  lstm (LSTM)                 (None, 16)                7488

#  dense (Dense)               (None, 1)                 17

# 임베딩 3.
# model.add(Embedding(input_dim=2, output_dim=100,))
# input_dim 은 더 많은 단어 사전의 개수(말뭉치의 개수)를 의미해서
# 많거나 적거나 상관없이 모델은 돌아간다. 단, 적을 경우 단어 사전의 개수를 적게 참고하겠다는 의미.
# 단 input_dim 적게: 단어사전으 줄여버리니까 성능 저하.

# 임베딩 4.
# model.add(Embedding(31,100))

# 임베딩 5.
model.add(Embedding(31,100, input_length=1))
# input_length = 1 =>> Non과 비슷한 의미로 warning은 나오나 모델은 돌아간다
# 즉, found shape=(None, 5) 형태일때=> input_length은 아예 명시x, 5, 1 때 모델은 돌아간다.


model.add(LSTM(16))
# model.add(BatchNormalization())
# model.add(Dense(20,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
# model.add(BatchNormalization())
model.add(Dense(1,activation='sigmoid'))

# model.summary()
# exit()
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
from keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=60,verbose=1,
                   restore_best_weights= True)
hist = model.fit(x_train,y_train, epochs= 100, batch_size= 2, callbacks=[es])

#4. 평가 예측
loss = model.evaluate(x_test,y_test,verbose=1)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
result =  (result > 0.5).astype(int)
print('loss :',loss[0])
print('acc :',loss[1])
print('result:',result)
predict = model.predict(padding_pred) #원래의 y값과 예측된 y값의 비교
# predict =  (predict > 0.5).astype(int)
print(predict)