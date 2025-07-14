#https://dacon.io/competitions/open/235576/overview/description

import numpy as np      #1.23.0
import pandas as pd     #2.2.3

# print(np.__version__) 
# print(pd.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1 data
# train_csv = pd.read_csv('.\_data\dacon\따릉이\train.csv')
## get rid of index column and just get the data

path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
print(train_csv) #[1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
print(test_csv) #[715 rows x 9 columns]

submission_csv = pd.read_csv(path+ 'submission.csv', index_col=0)
print(submission_csv) # [715 rows x 1 columns]

print(train_csv.shape)
print(test_csv.shape)
print(submission_csv.shape)


#columns very important
#### columns info describe
print(train_csv.columns)
#Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

print(train_csv.info())

#there are missing columns, so need to fill them out
#delete or modify
#######
# Data columns (total 10 columns):
#  #   Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64


                                                                                                            

# print(train_csv.describe())

###################################### 결측치 처리 1. 삭제 ###########################
#checks for missing values, and then prints out how many missing values there are in each column
# print(train_csv.isnull().sum()) #결측치의 개수 출력
# None
# hour                        0
# hour_bef_temperature        2
# hour_bef_precipitation      2
# hour_bef_windspeed          9
# hour_bef_humidity           2
# hour_bef_visibility         2
# hour_bef_ozone             76
# hour_bef_pm10              90
# hour_bef_pm2.5            117
# count                       0


###getting rid of the missing value
# train_csv = train_csv.dropna()
##### isnull and isna are the same
# print(train_csv.isnull().sum())
# print(train_csv.info())
# print(train_csv)    #[1328 rows x 10 columns]

# None
# hour                      0
# hour_bef_temperature      0
# hour_bef_precipitation    0
# hour_bef_windspeed        0
# hour_bef_humidity         0
# hour_bef_visibility       0
# hour_bef_ozone            0
# hour_bef_pm10             0
# hour_bef_pm2.5            0
# count                     0

 #   Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    1328 non-null   int64
#  1   hour_bef_temperature    1328 non-null   float64
#  2   hour_bef_precipitation  1328 non-null   float64
#  3   hour_bef_windspeed      1328 non-null   float64
#  4   hour_bef_humidity       1328 non-null   float64
#  5   hour_bef_visibility     1328 non-null   float64
#  6   hour_bef_ozone          1328 non-null   float64
#  7   hour_bef_pm10           1328 non-null   float64
#  8   hour_bef_pm2.5          1328 non-null   float64
#  9   count                   1328 non-null   float64


###################결측치 처리 2. 평균값 넣기#########################

#
train_csv = train_csv.fillna(train_csv.mean())
print(train_csv.isna().sum())
print(train_csv.info())



#################################test has missing values##########################

# dont drop the test data!!!!!!!!! DONT DROP
print(test_csv.info())
#  #   Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    715 non-null    int64
#  1   hour_bef_temperature    714 non-null    float64
#  2   hour_bef_precipitation  714 non-null    float64
#  3   hour_bef_windspeed      714 non-null    float64
#  4   hour_bef_humidity       714 non-null    float64
#  5   hour_bef_visibility     714 non-null    float64
#  6   hour_bef_ozone          680 non-null    float64
#  7   hour_bef_pm10           678 non-null    float64
#  8   hour_bef_pm2.5          679 non-null    float64

test_csv = test_csv.fillna(train_csv.mean())
print(test_csv.info())

 #   Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    715 non-null    int64
#  1   hour_bef_temperature    715 non-null    float64
#  2   hour_bef_precipitation  715 non-null    float64
#  3   hour_bef_windspeed      715 non-null    float64
#  4   hour_bef_humidity       715 non-null    float64
#  5   hour_bef_visibility     715 non-null    float64
#  6   hour_bef_ozone          715 non-null    float64
#  7   hour_bef_pm10           715 non-null    float64
#  8   hour_bef_pm2.5          715 non-null    float64

#drop() can delete columns or rows
#this means copy the remaining 9 
#axis =1 columns - mo0ves left,right
#axis =0 rows - moves up and down

x = train_csv.drop(['count'], axis=1)  #count라는 axis=1 열 삭제, 행은 axis =0
print(x) #[1459 rows x 9 columns]
# y = train_csv.

#take only the count colum to y 
y = train_csv['count'] 
print(y) #(1459,)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state= 333)

model = Sequential()
model.add(Dense(77, input_dim =9, activation='relu'))
model.add(Dense(77, activation='relu'))
model.add(Dense(77, activation='relu'))
model.add(Dense(77))
model.add(Dense(77))
model.add(Dense(1))


#3 compile and train
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 500, batch_size = 16)

#4 evaluate and predict
loss = model.evaluate(x_test, y_test)
print("loss:", loss)
result = model.predict(x_test)
# print("prediction:", result)

r2 = r2_score(y_test, result)
print("mean r2 score:",  r2)

rmse = np.sqrt(mean_squared_error(y_test, result))
print("RMSE result:", rmse)

#######결측치 평균값 넣은 결과######## epochs 400 batch 32 train0.8
#mean r2 score: 0.5692237798632227


#submission.csv에 test_csv의 예측값 넣기

y_submit = model.predict(test_csv) 
#train데이터의 shape와 동일한 컬럼을 확인하고 넣어.
                        #x_train.shape:(N, 9)
print(y_submit.shape)    #(715,1)



#####################submission.csv file 만들기// count 컬럼값만 넣어주기

submission_csv['count'] = y_submit
print(submission_csv)   


#####################
submission_csv.to_csv(path + 'submissionm_0521_1543.csv')
#            count
# id
# 0      96.123688
# 1     231.008057
# 2      78.424088
# 4     106.454376
# 5     111.288383
# ...          ...
# 2148   90.764603
# 2149   54.244217
# 2165  130.968597
# 2166  186.642456
# 2177  112.327148                   
                        


