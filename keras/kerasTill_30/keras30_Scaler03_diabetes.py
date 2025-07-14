#copied from 18-3
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1 data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(442, 10) (442,)

# exit()

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.75, random_state=333)



# scaler = MinMaxScaler()
# loss: 2893.813720703125
# r2 result: 0.4831635701805539

# scaler = StandardScaler()
# loss: 3314.853271484375
# r2 result: 0.40796565764850556

scaler = MaxAbsScaler()
# loss: 2788.070068359375
# r2 result: 0.5020494228161769

# scaler = RobustScaler()
# loss: 2987.388916015625
# r2 result: 0.4664509712472493


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = Sequential()
model.add(Dense(64, input_dim=10, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))  # regression output



model.compile(loss ='mse',optimizer = 'adam')

from keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 30,
    restore_best_weights = True,
)


hist = model.fit(x_train, y_train, epochs = 300, batch_size = 2,
          validation_split = 0.2,
          callbacks = [es])





loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)

r2 = r2_score(y_test, result)
print("loss:", loss)
print("r2 result:", r2)
