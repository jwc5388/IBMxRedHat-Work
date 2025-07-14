#copied from 18-4


#https://dacon.io/competitions/open/235576/overview/description

import numpy as np      #1.23.0
import pandas as pd     #2.2.3

from keras.models import Sequential, load_model
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1 data
# train_csv = pd.read_csv('.\_data\dacon\따릉이\train.csv')
## get rid of index column and just get the data

path = 'Study25/_data/dacon/ddarung/'
path_save = 'Study25/_data/dacon/ddarung/csv_files/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
# print(train_csv) #[1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
# print(test_csv) #[715 rows x 9 columns]

submission_csv = pd.read_csv(path+ 'submission.csv', index_col=0)
# print(submission_csv) # [715 rows x 1 columns]

train_csv = train_csv.dropna()


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



path_mcp = 'Study25/_save/keras28_mcp/04_dacon_dddarung/'
model = load_model(path + 'keras28_dacon_ddarung_save.h5')#3 compile and train


model.compile(loss = 'mse', optimizer = 'adam')


#4 evaluate and predict
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

result = model.predict(x_test)
# print("prediction:", result)

r2 = r2_score(y_test, result)
print("delete r2 score:",  r2)

rmse = np.sqrt(mean_squared_error(y_test, result))
print("delete RMSE result:", rmse)


y_submit = model.predict(test_csv) 



# #####################submission.csv file 만들기// count 컬럼값만 넣어주기

submission_csv['count'] = y_submit
# print(submission_csv)   


# #####################
submission_csv.to_csv(path + 'submission_0526_1300.csv')


