#copy from 26-3

import sklearn as sk
import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint


print(data)
x = data
y = target


print(x.shape, y.shape)

# (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, 
    random_state= 42,
    
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


model = Sequential()

model.add(Dense(32, input_dim = 13, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# model.summary()

model.summary()

path = 'Study25/_save/keras28_mcp/01_boston/'
model.save(path + 'keras28_boston_save.h5')

model.compile(loss = 'mse', optimizer= 'adam')

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 20,
    restore_best_weights = True,
)

# path= 'Study25/_save/keras28_mcp/01_boston'
############# mcp save file name 만들기 ##############
import datetime 
date = datetime.datetime.now()
print(date)  #2025-06-02 13:00:52.507403
print(type(date))   #<class 'datetime.datetime'>
#시간을 string으로 만들어라
date = date.strftime("%m%d_%H%M")
print(date) #0602_1305
print(type(date))   # <class 'str'>

# path= 'Study25/_save/keras27_mcp2/'

#hist 에서 제공되는 epoch, val_loss
filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = "".join([path, 'k28_', date, '_', filename])

print(filepath)

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    filepath = filepath
)



##list 형식으로 저장. 매 epoch 끝날 때마다 하나씩 들어가니, epoch 갯수만큼 
hist = model.fit(x_train,y_train, epochs = 100000, batch_size =1,
          verbose = 1,
          validation_split = 0.2,
          callbacks = [es, mcp],
          
          )


loss= model.evaluate(x_test,y_test)
print("loss:", loss)


result = model.predict(x_test)
r2 = r2_score(y_test, result)
print("r2 score", r2)
rmse = np.sqrt(mean_squared_error(y_test, result))
print()

# loss: 7.821259021759033
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
# r2 score 0.8933471266617676