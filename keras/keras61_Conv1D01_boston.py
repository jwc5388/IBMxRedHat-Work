#copy from 26-3

import sklearn as sk
import pandas as pd
import numpy as np
import time

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, LSTM, Reshape, Conv1D, Flatten
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

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# print(np.min(x_train), np.max(x_train))
# print(np.min(x_test), np.max(x_test))

x_train = x_train.reshape(-1,13,1)
x_test = x_test.reshape(-1,13,1)

# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)




#2 model 
model = Sequential([
    Conv1D(filters = 64, input_shape = (13,1), padding = 'same', kernel_size = 2),
    Conv1D(filters = 64, kernel_size = 2),
    Flatten(),
    Dense(units=16, activation='relu'),
    Dropout(0.2),
    Dense(units=16, input_shape=(16,)),
    Dense(units=1),
    
])

model.summary()
# model.summary()

model.summary()

path = 'Study25/_save/keras28_mcp/01_boston/'
model.save(path + 'keras28_boston_save.h5')

model.compile(loss = 'mse', optimizer= 'adam', metrics = ['mae'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 20,
    restore_best_weights = True,
)

############# mcp save file name 만들기 ##############
import datetime 
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = "".join([path, 'k28_', date, '_', filename])

print(filepath)

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    filepath = filepath
)


start = time.time()
##list 형식으로 저장. 매 epoch 끝날 때마다 하나씩 들어가니, epoch 갯수만큼 
hist = model.fit(x_train,y_train, epochs = 200, batch_size =32,
          verbose = 1,
          validation_split = 0.2,
          callbacks = [es, mcp],
          
          )

end = time.time()



loss= model.evaluate(x_test,y_test)
print("loss:", loss)

result = model.predict(x_test)
r2 = r2_score(y_test, result)
print("r2 score", r2)
rmse = np.sqrt(mean_squared_error(y_test, result))
print("rmse:", rmse)

# 4. 출력
print("\n📊 Final Evaluation:")
print(f"✅ Loss (MSE): {loss:.4f}")
print(f"✅ RMSE      : {rmse:.4f}")
print(f"✅ R² Score  : {r2:.4f}")
print(f"⏱️ Training Time: {end - start:.2f}초")


# 📊 Final Evaluation:
# ✅ Loss (MSE): 34.1673
# ✅ RMSE      : 5.8453
# ✅ R² Score  : 0.5341
# ⏱️ Training Time: 30.83초