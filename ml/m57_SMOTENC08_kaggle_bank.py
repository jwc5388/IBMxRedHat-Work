# import numpy as np
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, BatchNormalization
# from keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import classification_report
# from sklearn.utils import class_weight
# from imblearn.over_sampling import SMOTENC
# import os

# # 1. 경로 설정
# if os.path.exists('/workspace/TensorJae/Study25/'):
#     BASE_PATH = '/workspace/TensorJae/Study25/'
# else:
#     BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
# basepath = os.path.join(BASE_PATH)

# # 2. 데이터 로드
# path = basepath + '_data/kaggle/bank/'
# train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# # 3. 범주형 인코딩
# le_geo = LabelEncoder()
# le_gender = LabelEncoder()
# train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
# train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])
# test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
# test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

# # 4. 불필요 컬럼 제거
# train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
# test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# # 5. 피처/타깃 분리
# x = train_csv.drop(['Exited'], axis=1)
# y = train_csv['Exited']

# # 6. SMOTENC (범주형 인덱스: Geography(1), Gender(2), NumOfProducts(6), HasCrCard(7), IsActiveMember(8))
# smotenc = SMOTENC(random_state=337, categorical_features=[1, 2, 6, 7, 8])
# x_res, y_res = smotenc.fit_resample(x, y)

# # 7. 스케일링
# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(x_res)
# test_scaled = scaler.transform(test_csv)
# y = y_res  # y 업데이트

# # 8. 학습/평가 데이터 분리
# x_train, x_test, y_train, y_test = train_test_split(
#     x_scaled, y, train_size=0.8, random_state=333, stratify=y
# )

# # 9. 클래스 가중치 계산
# weights = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y_train),
#     y=y_train
# )
# class_weights = dict(enumerate(weights))

# # 10. 모델 정의
# model = Sequential([
#     Dense(128, input_dim=10, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),
#     Dense(256, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),
#     Dense(128, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# # 11. 컴파일
# model.compile(
#     loss='binary_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )

# # 12. 콜백
# es = EarlyStopping(
#     monitor='val_loss',
#     mode='min',
#     patience=20,
#     restore_best_weights=True
# )

# # 13. 학습
# hist = model.fit(
#     x_train, y_train,
#     epochs=200,
#     batch_size=64,
#     validation_split=0.2,
#     callbacks=[es],
#     class_weight=class_weights,
#     verbose=1
# )

# # 14. 평가
# loss, acc = model.evaluate(x_test, y_test)
# print(f"\n✅ Test Accuracy: {acc:.4f}")

# # 15. 리포트
# y_pred = model.predict(x_test)
# y_pred_binary = np.round(y_pred).astype(int)
# print("\nClassification Report:\n", classification_report(y_test, y_pred_binary))

# # 16. 제출용 예측 및 저장
# y_submit = model.predict(test_scaled)
# submission_csv['Exited'] = y_submit
# submission_filename = path + 'submission_0527_improved.csv'
# submission_csv.to_csv(submission_filename)
# print(f"✅ Submission file saved: {submission_filename}")



# https://www.kaggle.com/competitions/playground-series-s4e1/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import class_weight
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib
from imblearn.over_sampling import SMOTENC

import os
if os.path.exists('/workspace/TensorJae/Study25/'):
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
basepath = os.path.join(BASE_PATH)

# 2. 데이터 로드
path = basepath + '_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
le_geo = LabelEncoder() # 클래스를 인스턴스 한다.
le_gender = LabelEncoder()
#print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited'],

train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])
print(train_csv['Geography'])
print(train_csv['Geography'].value_counts()) #pandas
# 0    94215
# 2    36213
# 1    34606
print(train_csv['Gender'].value_counts()) #pandas np.unique(data, return_counts=True)
# 1    93150
# 0    71884
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

train_csv = train_csv.drop(['CustomerId','Surname'],axis= 1)
test_csv = test_csv.drop(['CustomerId','Surname'], axis= 1)
# print(train_csv.columns)
# print(test_csv)

train_csv['Balance'] = train_csv['Balance'].replace(0, train_csv['Balance'].mean())
test_csv['Balance'] = test_csv['Balance'].replace(0, test_csv['Balance'].mean())

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']
# print(x.columns)
# Index(['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
#        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'],
#       dtype='object')

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.1, random_state= 42)

standard = StandardScaler() # 표준화
scaler = MinMaxScaler() # 정규화

x_train[['CreditScore','Age','Tenure','Balance','EstimatedSalary']] = standard.fit_transform(x_train[['CreditScore','Age','Tenure','Balance','EstimatedSalary']])        # train 데이터에 맞춰서 스케일링
x_test[['CreditScore','Age','Tenure','Balance','EstimatedSalary']]= standard.transform(x_test[['CreditScore','Age','Tenure','Balance','EstimatedSalary']]) # test 데이터는 transform만!
test_csv[['CreditScore','Age','Tenure','Balance','EstimatedSalary']] = standard.transform(test_csv[['CreditScore','Age','Tenure','Balance','EstimatedSalary']])

smotenc = SMOTENC(random_state=337,
                   categorical_features=[1,2,4,5,],
                   )
x_res,y_res = smotenc.fit_resample(x_train,y_train)

# x_train[['EstimatedSalary']] = scaler.fit_transform(x_train[['EstimatedSalary']])        # train 데이터에 맞춰서 스케일링
# x_test[['EstimatedSalary']]= scaler.transform(x_test[['EstimatedSalary']]) # test 데이터는 transform만!
# test_csv[['EstimatedSalary']] = scaler.transform(test_csv[['EstimatedSalary']])

#7. Compute class weights
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))

# 2. 모델 구조
model = Sequential()
model.add(Dense(256, input_dim = 10, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(4, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid')) 


# 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss', # 평가 지표로 확인하겠다
    mode= 'min', # 최대값 max, 알아서 찾아줘:auto
    patience=20, # 10번까지 초과해도 넘어가겠다
    restore_best_weights=True, # val_loss값이 가장 낮은 값으로 저장 해놓겠다(False시 => 최소값 이후 10번째 값으로 그냥 잡는다.)
)
#model.compile(loss='mse', optimizer='adam')
model.compile(loss='mse', optimizer='adam', metrics=['acc']) # 이진 분류 loss는 무조건
hist = model.fit(x_res, y_res,epochs= 50, batch_size= 12,verbose=2,validation_split=0.1, callbacks=[es],class_weight=class_weights,)

# 4. 평가 예측
results = model.evaluate(x_test, y_test)
print(results) 
#[0.034758370369672775, 0.9824561476707458]

#print('loss = ',results[0])
#print('acc = ', round(results[1],4)) # 반올림
y_predict = model.predict(x_test)
y_predict =  (y_predict > 0.5).astype(int)
from sklearn.metrics import accuracy_score # 이진만 받을 수 있다
accuracy_score = accuracy_score(y_test, y_predict)

# submission.csv에 test_csv의 예측값 넣기
y_submit = model.predict(test_csv)
print(y_submit)
#y_submit =  (y_submit > 0.5).astype(int)
#y_pred = [1 if y > 0.5 else 0 for y in y_submit]
######## submission.csv 파일 만들기 //count컬럼값만 넣어주기########
submission_csv['Exited'] = y_submit
#print(submission_csv)
