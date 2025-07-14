#https://dacon.io/competitions/official/236068/overview/description


import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint




path = 'Study25/_data/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col= 0)

print(train_csv.info()) # [652 rows x 9 columns]
print(test_csv.info()) #  [116 rows x 8 columns]
print(sample_submission_csv.info()) # [116 rows x 1 columns]

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())

print(x)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state= 33)


#2 model


path_mcp = 'Study25/_save/keras28_mcp/07_dacon_diabetes/'
model =load_model(path_mcp + 'keras28_dacon_diabetes.h5')



#3 compile and train
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])


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