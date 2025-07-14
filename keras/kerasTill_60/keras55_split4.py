from os import times
from matplotlib.pylab import RandomState
import numpy as np
from xgboost import train

a = np.array(range(1,101))
x_predict = np.array(range(96,106))

timesteps = 11

# x = (N,10,1).  -> (N,5,2)
# y = (N,1)     -> 


def  split_xy(dataset, timesteps):
    all = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i+timesteps)]
        all.append(subset)
    all = np.array(all)
    x = all[:,:-1]
    y = all[:,-1]
    return x,y

x, y = split_xy(a, 11)

print(x.shape)

# def split_pred(dataset, timesteps):
#     all = []
#     for i in range(len(dataset) - timesteps +1):
#         subset = dataset[i: (i+timesteps)]
#         all.append(subset)
#     all = np.array(all[:])
#     return all
# x_predict = split_pred(x_predict, 10)


print(x_predict.shape)



x = x.reshape(-1,5,2)

x_predict = x_predict.reshape(-1,5,2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=42
)


from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM, GRU

model = Sequential()
model.add(SimpleRNN(128, input_shape = (5,2), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))


model.compile(loss= 'mse', optimizer = 'adam', metrics = ['acc'])

model.fit(x_train, y_train, epochs = 500, batch_size = 32, validation_split=0.2, verbose =1)

loss = model.evaluate(x_test, y_test)
result = model.predict(x_predict)

print('loss:', loss)
print('prediction:', np.round(result,4))


# loss: [0.003057911992073059, 0.0]
# prediction: [[106.0478]]



# loss: [0.00563559727743268, 0.0]
# prediction: [[106.1051]]