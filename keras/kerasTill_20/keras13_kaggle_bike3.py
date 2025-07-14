#train.csv와 new_test.csv로 count 예측하면 됨

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

path = 'Study25/_data/kaggle/bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'new_test.csv')
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(train_csv.info)
# [10886 rows x 11 columns]>
print(test_csv.info)
# [6493 rows x 10 columns]>

# exit()

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=333)

model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))  # input_dim matches your 8 features
model.add(Dense(64, activation='relu'))   # Reduced complexity to prevent overfitting
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))


model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs = 100, batch_size = 16)

loss = model.evaluate(x_test,y_test)
result = model.predict(x_test)

y_submit = model.predict(test_csv)
submission_csv['count'] = y_submit

submission_csv.to_csv(path + 'submission_0522_1843.csv')

