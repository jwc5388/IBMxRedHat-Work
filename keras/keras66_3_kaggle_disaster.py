# # https://www.kaggle.com/competitions/nlp-getting-started/overview

# import numpy as np
# import pandas as pd

# from sklearn import feature_extraction, linear_model, model_selection, preprocessing
# from keras.preprocessing.text import Tokenizer
# from keras.layers import Dense, Dropout, BatchNormalization, LSTM, Conv1D, Embedding, GlobalAveragePooling1D, Bidirectional
# from keras.models import Sequential
# from keras.preprocessing.sequence import pad_sequences
# from keras.callbacks import EarlyStopping
# import datetime

# import warnings
# warnings.filterwarnings('ignore')

# # 파일 저장을 위한 타임스탬프 경로 설정
# timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
# # 데이터 로드
# path = './Study25/_data/kaggle/nlp_disaster/'
# train_csv = pd.read_csv(path + 'train.csv')
# test_csv = pd.read_csv(path + 'test.csv')
# submission_csv = pd.read_csv(path + 'sample_submission.csv')

# # print(f"훈련 데이터 : {train.shape}")
# # print(f"테스트 데이터 : {test.shape}")
# # 훈련 데이터 : (7613, 5)
# # 테스트 데이터 : (3263, 4)

# def merge_text(df):
#     df['keyword'] = df['keyword'].fillna('unknown')
#     df['location'] = df['location'].fillna('unknown').str.lower()
#     df['text'] = df['text'] + ' keyword: ' + df['keyword'] + \
#         ' location: ' + df['location']
#     return df

# train = merge_text(train_csv)
# test = merge_text(test_csv)

# # print(train[:100]) #[7613 rows x 5 columns]
# # print(test) #[3263 rows x 4 columns]
# print(train['text'].iloc[95]) 
# # 9 Mile backup on I-77 South...accident blocking the Right 2 Lanes at Exit 31 Langtree Rd...

# # Tokenizer
# token = Tokenizer()
# token.fit_on_texts(train['text'])

# x = token.texts_to_sequences(train['text'])
# x_test = token.texts_to_sequences(test['text'])



# # 시퀀스 길이 확인
# max_len = max(len(seq) for seq in x)
# print("최대 시퀀스 길이:", max_len)

# print('disaster 최대길이 :', max(len(i) for i in x))  
# print('disaster 최소길이 :', min(len(i) for i in x))  
# print('disaster 평균길이 :', sum(map(len, x))/len(x))  

# # Padding
# x_train_pad = pad_sequences(x, maxlen=100, padding='post', truncating='post')
# x_test_pad = pad_sequences(x_test, maxlen=100, padding='post', truncating='post')
# y_train = train['target'].values

# # Train/Validation 분리
# from sklearn.model_selection import train_test_split
# x_train_final, x_val, y_train_final, y_val = train_test_split(
#     x_train_pad, y_train, test_size=0.2, random_state=42
# )

# # 모델 구성
# model = Sequential([
#     Embedding(input_dim=10000, output_dim=128, input_length=100),
#     Bidirectional(LSTM(64, return_sequences=False)),
#     Dropout(0.5),
#     Dense(64, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# es = EarlyStopping(
#     monitor = 'val_loss',
#     mode = 'min',
#     restore_best_weights=True,
#     patience=15
# )

# # 모델 학습
# model.fit(x_train_final, y_train_final, epochs=50, batch_size=32, validation_data=(x_val, y_val), callbacks=[es])

# # ✅ 모델 성능 평가
# loss, acc = model.evaluate(x_val, y_val)
# print(f"검증 손실(loss): {loss:.4f}")
# print(f"검증 정확도(accuracy): {acc:.4f}")

# # 예측 및 제출
# y_test_pred = (model.predict(x_test_pad) > 0.5).astype(int).flatten()

# submission_csv['target'] = y_test_pred
# submission_csv.to_csv(path + f'submission_tokenizer_dnn_{timestamp}.csv', index=False)
# print(f"제출 파일 저장 완료: submission_tokenizer_dnn_{timestamp}.csv")



#####keyword, location 은 onehot, text = embedding 후 이 셋 concat

import numpy as np
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Bidirectional, Dropout, Dense, Concatenate
from keras.callbacks import EarlyStopping

# 1. 경로 및 파일 불러오기
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
path = './Study25/_data/kaggle/nlp_disaster/'
train_csv = pd.read_csv(path + 'train.csv')
test_csv = pd.read_csv(path + 'test.csv')
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 2. 결측치 처리
for df in [train_csv, test_csv]:
    df['keyword'] = df['keyword'].fillna('unknown')
    df['location'] = df['location'].fillna('unknown').str.lower()
    df['text'] = df['text'].astype(str)
    
# 3. Tokenizer - text 처리
token = Tokenizer(num_words=10000, oov_token="<OOV>")
token.fit_on_texts(train_csv['text'])

x_text = token.texts_to_sequences(train_csv['text'])
x_text_test = token.texts_to_sequences(test_csv['text'])

maxlen = 100
x_text_pad = pad_sequences(x_text, maxlen=maxlen, padding='post', truncating='post')
x_text_test_pad = pad_sequences(x_text_test, maxlen=maxlen, padding='post', truncating='post')

# 4. One-hot encoding - keyword, location
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

x_keyword = ohe.fit_transform(train_csv[['keyword']])
x_keyword_test = ohe.transform(test_csv[['keyword']])

x_location = ohe.fit_transform(train_csv[['location']])  # 새로 학습
x_location_test = ohe.transform(test_csv[['location']])

# 5. Label
y_train = train_csv['target'].values

# 6. Train/Validation Split
x_text_train, x_text_val, x_key_train, x_key_val, x_loc_train, x_loc_val, y_train_final, y_val = train_test_split(
    x_text_pad, x_keyword, x_location, y_train, test_size=0.2, random_state=42
)

# 7. 모델 구성 (Functional API)
# 입력층
text_input = Input(shape=(maxlen,))
keyword_input = Input(shape=(x_key_train.shape[1],))
location_input = Input(shape=(x_loc_train.shape[1],))

# text branch
x = Embedding(input_dim=10000, output_dim=128, input_length=maxlen)(text_input)
x = Bidirectional(LSTM(64))(x)
x = Dropout(0.5)(x)

# 합치기
concat = Concatenate()([x, keyword_input, location_input])
dense = Dense(64, activation='relu')(concat)
output = Dense(1, activation='sigmoid')(dense)

# 모델 정의
model = Model(inputs=[text_input, keyword_input, location_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# 8. 학습
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(
    [x_text_train, x_key_train, x_loc_train],
    y_train_final,
    validation_data=([x_text_val, x_key_val, x_loc_val], y_val),
    epochs=50,
    batch_size=32,
    callbacks=[es]
)

# 9. 평가
loss, acc = model.evaluate([x_text_val, x_key_val, x_loc_val], y_val)
print(f"검증 손실: {loss:.4f}")
print(f"검증 정확도: {acc:.4f}")

# 10. 예측 및 제출
y_test_pred = (model.predict([x_text_test_pad, x_keyword_test, x_location_test]) > 0.5).astype(int).flatten()
submission_csv['target'] = y_test_pred
submission_csv.to_csv(path + f'submission_combined_{timestamp}.csv', index=False)
print(f"제출 파일 저장 완료: submission_combined_{timestamp}.csv")


# 검증 손실: 0.4347
# 검증 정확도: 0.8129
# 102/102 [==============================] - 1s 8ms/step
# 제출 파일 저장 완료: submission_combined_20250630_1808.csv