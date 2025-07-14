import numpy as np

a = np.array(range(1,101))
x_pred = np.array(range(96,106))
# print(x_pred.shape)

timesteps = 6

def split_1d(dataset, timesteps):
    all = []
    for i in range(len(dataset) - timesteps + 1) :
        subset = dataset[i : (i+timesteps)]
        all.append(subset)
    all = np.array(all)
    x = all[:,:-1]
    y = all[:,-1]
    return x, y


# x = all[:,:-1]   # All rows, except last column
# y = all[:,-1]    # Last column only

x, y = split_1d(a, 6)
# print(x, y)
# print(x.shape, y.shape) #(95, 5) (95,)

def split_all(dataset, timesteps):
    all = []
    for i in range(len(dataset) - timesteps + 1) :
        subset = dataset[i : (i+timesteps)]
        all.append(subset)
    all = np.array(all[:])
    return all
x_pred = split_all(x_pred, 5)
# print(x_pred)
# [[ 96  97  98  99 100]
#  [ 97  98  99 100 101]
#  [ 98  99 100 101 102]
#  [ 99 100 101 102 103]
#  [100 101 102 103 104]
#  [101 102 103 104 105]]

# exit()

x = x.reshape(-1,5,1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM, GRU
model = Sequential()
model.add(SimpleRNN(128, input_shape=(5,1), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)

loss = model.evaluate(x_test, y_test)
results = model.predict(x_pred)

print('loss: ', np.round(loss[0],4))
print('예측값:', np.round(results,4))


# loss:  0.0
# 예측값: [[100.9987]
#  [101.9988]
#  [102.999 ]
#  [103.999 ]
#  [104.9992]
#  [105.9992]]


