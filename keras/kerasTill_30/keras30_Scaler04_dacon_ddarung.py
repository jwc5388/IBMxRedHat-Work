
#https://dacon.io/competitions/open/235576/overview/description

import numpy as np      #1.23.0
import pandas as pd     #2.2.3

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


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


# scaler = MinMaxScaler()
# loss: 1810.567138671875
# r2 score: 0.7403217732190246
# RMSE result: 42.55076139297194

# scaler = StandardScaler()
# loss: 2217.904296875
# r2 score: 0.6818999706659208
# RMSE result: 47.09463353236422


scaler = MaxAbsScaler()
# loss: 1777.40966796875
# r2 score: 0.7450773643146207
# RMSE result: 42.15933648688444


# scaler = RobustScaler()
# loss: 2007.0225830078125
# r2 score: 0.7121454011421071
# RMSE result: 44.79980806235708

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)



model = Sequential()
model.add(Dense(77, input_dim =9, activation='relu'))
model.add(Dense(77, activation='relu'))
model.add(Dense(77))
model.add(Dense(77))
model.add(Dense(77))
model.add(Dense(1))

#3 compile and train
model.compile(loss = 'mse', optimizer = 'adam')

from keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 30,
    restore_best_weights = True,
)


hist = model.fit(x_train, y_train, epochs = 300 , batch_size = 16,
         validation_split = 0.2 ,
         callbacks = [es])


#4 evaluate and predict
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

result = model.predict(x_test)
# print("prediction:", result)

r2 = r2_score(y_test, result)
print("r2 score:",  r2)

rmse = np.sqrt(mean_squared_error(y_test, result))
print("RMSE result:", rmse)


y_submit = model.predict(test_csv) 



# #####################submission.csv file 만들기// count 컬럼값만 넣어주기

# submission_csv['count'] = y_submit
# # print(submission_csv)   


# # #####################
# submission_csv.to_csv(path + 'submission_0526_1300.csv')


