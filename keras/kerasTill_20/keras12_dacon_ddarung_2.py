#https://dacon.io/competitions/open/235576/overview/description

import numpy as np      #1.23.0
import pandas as pd     #2.2.3

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1 data
# train_csv = pd.read_csv('.\_data\dacon\따릉이\train.csv')
## get rid of index column and just get the data

path = 'Study25/_data/dacon/ddarung/'
path_save = 'Study25/_data/dacon/ddarung/csv_files/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
print(train_csv) #[1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
print(test_csv) #[715 rows x 9 columns]

submission_csv = pd.read_csv(path+ 'submission.csv', index_col=0)
print(submission_csv) # [715 rows x 1 columns]

# print(train_csv.shape)
# print(test_csv.shape)
# print(submission_csv.shape)

# print(train_csv.info())


###getting rid of the missing value
train_csv = train_csv.dropna()
##### isnull and isna are the same
# print(train_csv.isnull().sum())
# print(train_csv.info())
# print(train_csv)    #[1328 rows x 10 columns]

###################결측치 처리 2. 평균값 넣기#########################

#
# train_csv = train_csv.fillna(train_csv.mean())
# print(train_csv.isna().sum())
# print(train_csv.info())


# test_csv = test_csv.dropna()
test_csv = test_csv.fillna(train_csv.mean())
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)  #count라는 axis=1 열 삭제, 행은 axis =0
print(x) #[1459 rows x 9 columns]
# y = train_csv.

#take only the count colum to y 
y = train_csv['count'] 
print(y) #(1459,)



x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state= 3333)

model = Sequential()
model.add(Dense(77, input_dim =9, activation='relu'))
model.add(Dense(77, activation='relu'))
model.add(Dense(77, activation='relu'))
model.add(Dense(77, activation='relu'))
model.add(Dense(77, activation='relu'))
model.add(Dense(1))


#3 compile and train
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 30, batch_size = 16)

#4 evaluate and predict
loss = model.evaluate(x_test, y_test)

result = model.predict(x_test)
# print("prediction:", result)

r2 = r2_score(y_test, result)
print("delete r2 score:",  r2)

rmse = np.sqrt(mean_squared_error(y_test, result))
print("delete RMSE result:", rmse)

print("loss:", loss)

# exit()


y_submit = model.predict(test_csv) 
#train데이터의 shape와 동일한 컬럼을 확인하고 넣어.
                        #x_train.shape:(N, 9)
print(y_submit.shape)    #(715,1)



#####################submission.csv file 만들기// count 컬럼값만 넣어주기

submission_csv['count'] = y_submit
print(submission_csv)   


#####################
submission_csv.to_csv(path + 'submissiond_0521_1600.csv')

# rmse = np.sqrt(mean_squared_error(y_test, result))
# print("RMSE result:", rmse)



#########결측치 삭제한 결과######################

# delete r2 score: 0.6009595033226571
# delete RMSE result: 54.33024778625692
# loss: 2951.776123046875


##after relu
# delete r2 score: 0.7184530279325239
# delete RMSE result: 45.636111240724446
# loss: 2082.654541015625
####기준 

# delete r2 score: 0.7541795804508532
# delete RMSE result: 41.39982877953518
# loss: 1713.9459228515625


# delete r2 score: 0.740600997706628
# delete RMSE result: 42.527878438374024
# loss: 1808.6204833984375
"""
r2 0.58 이상
loss 는 2400.0 이하
1. 전처리부분: 결측치제거, 결측치 평균으로 채우기 비교
    train size ???
    random_state
2. 하이퍼파라미터 튜닝부분:
    epochs
    batch_size
    모델구성(레이어, 노드)
    """


