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

path = 'Study25/_save/keras28_mcp/01_boston'

model = load_model(path + 'keras28_boston_save.h5')



model.summary()


model.compile(loss = 'mse', optimizer= 'adam')

#4 evaluate and predict

loss= model.evaluate(x_test,y_test)
print("loss:", loss)

result = model.predict(x_test)
r2 = r2_score(y_test, result)
print("r2 score", r2)
rmse = np.sqrt(mean_squared_error(y_test, result))
print()

