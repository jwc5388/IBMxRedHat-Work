
#1. train_csv 에서 casual과 registered 를 y로 잡는다
#2. 훈련해서, test_csv의 casual과 registered를 예측(predict) 한다.
#3. 예측한 casual 과 registered를 test_csv에 컬럼으로 넣는다.
#4. (N, 8) -> (N,10) test.csv 파일을 new_test.csv 파일을 만든다.

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

path = path = 'Study25/_data/kaggle/bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv.columns)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)


x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv[['casual', 'registered']]

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state= 333)


model1 = Sequential()
model1.add(Dense(128, input_dim = 8, activation = 'relu'))
model1.add(Dense(64, activation = 'relu'))
model1.add(Dense(64, activation = 'relu'))
model1.add(Dense(64, activation = 'relu'))
model1.add(Dense(32, activation = 'relu'))
model1.add(Dense(2, activation = 'linear'))


model1.compile(loss = 'mse', optimizer = 'adam')
model1.fit(x_train, y_train, epochs=100, batch_size = 16)

result = model1.predict(x_test)
r2 = r2_score(y_test,result)



# exit()

y_predict = model1.predict(test_csv)
test_csv[['casual', 'registered']] = y_predict
test_csv.to_csv(path + 'new_test.csv', index = False)

# test_csv[['casual', 'registered']] = y_predict
# test_csv['registerd'] = result_register.flatten()


# test_csv.to_csv(path + 'new_test.csv')
