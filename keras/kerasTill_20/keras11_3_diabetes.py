from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

#1 data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(442, 10) (442,)

# exit()

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.75, random_state=7)

model = Sequential()
model.add(Dense(64, input_dim=10))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))  # regression output

model.compile(loss ='mse',optimizer = 'adam')
model.fit(x_train, y_train, epochs = 300, batch_size = 2)

loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)

r2 = r2_score(y_test, result)
print("r2 result:", r2)

# r2 result: 0.5352923521648715


##over 0.62