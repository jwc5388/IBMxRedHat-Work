#https://dacon.io/competitions/official/236068/overview/description


import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score




path = 'Study25/_data/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col= 0)

print(train_csv.info()) # [652 rows x 9 columns]
print(test_csv.info()) #  [116 rows x 8 columns]
print(sample_submission_csv.info()) # [116 rows x 1 columns]

# print(train_csv.columns)

# exit()ㅠ ㅍ



x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())

print(x)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state= 33)


#2 model
model = Sequential()
model.add(Dense(100, input_dim = 8, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


#3 compile and train
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

es = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
                   patience = 30,
                   restore_best_weights = True
)

hist = model.fit(x_train, y_train, epochs = 200, batch_size = 2, validation_split = 0.2, verbose = 1, callbacks = [es])

result = model.evaluate(x_test, y_test)
print('acc:', result[1])

result1 = model.predict(x_test)

y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)

# accuracy = accuracy_score(y_test, y_submit)
# print('accuracy:', accuracy)



sample_submission_csv['Outcome'] = y_submit
print(sample_submission_csv.head)

sample_submission_csv.to_csv(path + 'submission_0527_1433.csv')

#acc: 0.7557252049446106