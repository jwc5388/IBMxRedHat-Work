from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.75, random_state=333)

model = Sequential()
model.add(Dense(64, input_dim=10, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))  # regression output

model.compile(loss ='mse',optimizer = 'adam')
hist = model.fit(x_train, y_train, epochs = 300, batch_size = 2,
          validation_split = 0.2)


# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6)) # 9x6 size 
# plt.plot(hist.history['loss'], c = 'red', label = 'loss') # y값만 넣으면 시간순으로 그림 그림
# plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')
# plt.title("Diabetes loss")
# plt.xlabel('epoch')
# plt.ylabel('loss') 
# plt.legend(loc='upper right') #우측 상단에 label 표시
# plt.grid() #격자 표시


# plt.show()


loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)

r2 = r2_score(y_test, result)
print("r2 result:", r2)

# r2 result: 0.5352923521648715


##over 0.62
# after val
# r2 result: 0.4915165622969718