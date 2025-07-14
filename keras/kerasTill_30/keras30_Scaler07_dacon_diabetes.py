#https://dacon.io/competitions/official/236068/overview/description


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint




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


# === Scale Data ===
scaler = MinMaxScaler()
# acc: 0.7786259651184082
# loss: 0.6667569875717163
# r2 score: 0.224845290184021

# scaler = StandardScaler()
# acc: 0.7557252049446106
# loss: 0.6298522353172302
# r2 score: 0.19405078887939453

# scaler = RobustScaler()
# acc: 0.7557252049446106
# loss: 0.5751410722732544
# r2 score: 0.23607760667800903

# scaler = MaxAbsScaler()
# acc: 0.7557252049446106
# loss: 0.5185191035270691
# r2 score: 0.19584524631500244


# x = scaler.fit_transform(x)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

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
print('loss:', result[0])

result1 = model.predict(x_test)
r2 = r2_score(y_test, result1)
print('r2 score:', r2)

y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)

# accuracy = accuracy_score(y_test, y_submit)
# print('accuracy:', accuracy)



# sample_submission_csv['Outcome'] = y_submit
# print(sample_submission_csv.head)

# sample_submission_csv.to_csv(path + 'submission_0527_1433.csv')

#acc: 0.7557252049446106