# copy from 26-6

import sklearn as sk
import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping
import keras.losses
from keras.models import load_model
import keras.losses

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


# model = Sequential()

# model.add(Dense(32, input_dim = 13, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1))


path = 'Study25/_save/keras27_mcp/'
model = load_model(path + 'keras27_mcp1.h5', custom_objects={'mse': keras.losses.MeanSquaredError()})
#이거 초기 가중치 - 그래서 구림
# model.load_weights(path + 'keras26_5_save1.weights.h5')

#훈련한 가중치!!!!!!!^#&!^&!^$ㅕㅎ쪼ㅕ홓ㄴ오
# model.load_weights(path + 'keras26_5_save2.weights.h5')

# model = load_model(
#     path + 'keras26_3_save.h5',
#     custom_objects={'mse': keras.losses.MeanSquaredError()}
# )


model.summary()


#3
model.compile(loss = 'mse', optimizer= 'adam')


# es = EarlyStopping(
#     monitor = 'val_loss',
#     mode = 'auto',
#     patience = 20,
#     restore_best_weights = True,
# )

# ##list 형식으로 저장. 매 epoch 끝날 때마다 하나씩 들어가니, epoch 갯수만큼 
# hist = model.fit(x_train,y_train, epochs = 100000, batch_size =1,
#           verbose = 1,
#           validation_split = 0.2,
#           callbacks = [es],
          
#           )


#4 evaluate and predict

loss= model.evaluate(x_test,y_test)
print("loss:", loss)

result = model.predict(x_test)
r2 = r2_score(y_test, result)
print("r2 score", r2)
rmse = np.sqrt(mean_squared_error(y_test, result))
print()



# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ dense (Dense)                        │ (None, 32)                  │             448 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 32)                  │           1,056 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_2 (Dense)                      │ (None, 32)                  │           1,056 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_3 (Dense)                      │ (None, 32)                  │           1,056 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_4 (Dense)                      │ (None, 32)                  │           1,056 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_5 (Dense)                      │ (None, 1)                   │              33 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 4,707 (18.39 KB)
#  Trainable params: 4,705 (18.38 KB)
#  Non-trainable params: 0 (0.00 B)
#  Optimizer params: 2 (12.00 B)
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 6.5551  
# loss: 7.821259021759033
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
# r2 score 0.8933471266617676