# import numpy as np
# import pandas as pd

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
# # https://www.kaggle.com/competitions/playground-series-s4e1/data
# from sklearn.metrics import accuracy_score
# import time

# #1 data
# path = 'Study25/_data/kaggle/bank/'

# train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
# test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
# submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col = 0)


# print(train_csv)
# print(train_csv.head()) #first few
# print(train_csv.tail()) # last few
# # print(train_csv.head(10)) #default = 5


# #########important######## 결측치 확인 ###############################
# print(train_csv.isna().sum())
# print(test_csv.isna().sum())

# print(train_csv.columns)
# # Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
# #        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
# #        'EstimatedSalary', 'Exited'],
# #       dtype='object')

# from sklearn.preprocessing import LabelEncoder
# ###for #문자 데이터 수치화!!! encoding categorial variables 

# #클래스를 정의한다 = instance화 한다
# le_geo = LabelEncoder()
# le_gender = LabelEncoder()
# train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
# train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])


# test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
# test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

# # train_csv['Geography'], unique_countries
# print(train_csv['Geography'])
# print(pd.value_counts(train_csv['Geography']))
# # Geography
# # 0    94215
# # 2    36213
# # 1    34606
# print(pd.value_counts(train_csv['Gender']))
# # Gender
# # 1    93150
# # 0    71884

# train_csv = train_csv.drop(['CustomerId', 'Surname'], axis = 1)
# test_csv = test_csv.drop(['CustomerId', 'Surname'],  axis = 1)

# print(train_csv.columns)

# # Index(['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
# #        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
# #        'Exited'],
# #       dtype='object')

# x = train_csv.drop(['Exited'], axis = 1)
# print(x.shape)
# y = train_csv['Exited']
# print(y.shape)


# x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state= 33)

# model = Sequential()
# model.add(Dense(128, input_dim = 10, activation = 'relu'))
# model.add(Dense(256, activation = 'relu'))
# model.add(Dense(256, activation = 'relu'))
# model.add(Dense(256, activation = 'relu'))
# model.add(Dense(256, activation = 'relu'))
# model.add(Dense(128, activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(1, activation = 'sigmoid'))


# #compile and train

# model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

# es = EarlyStopping(
#     monitor = 'val_loss',
#     mode = 'min',
#     patience = 20,
#     restore_best_weights = True,
# )

# model.fit(x_train, y_train, epochs = 100000, batch_size = 128, validation_split = 0.2, verbose = 1, callbacks = [es])

# result = model.evaluate(x_test, y_test)
# print("accuracy :", result[1] )

# y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit).astype(int)

# submission_csv['Exited'] = y_submit
# submission_csv.to_csv(path + 'submission_0527_1730.csv')




# # print(train_csv.info())
# # print(train_csv.info())
# # print(train_csv.info())


# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import accuracy_score

# # 1. Load Data
# path = 'Study25/_data/kaggle/bank/'

# train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# # 2. Check and encode categorical features
# le_geo = LabelEncoder()
# le_gender = LabelEncoder()

# train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
# train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])

# test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
# test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

# # 3. Drop unneeded columns
# train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
# test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# # 4. Separate features and target
# x = train_csv.drop(['Exited'], axis=1)
# y = train_csv['Exited']

# # 5. Scale features
# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(x)
# test_scaled = scaler.transform(test_csv)

# # 6. Train-test split (after scaling)
# x_train, x_test, y_train, y_test = train_test_split(
#     x_scaled, y, train_size=0.8, random_state=33
# )

# # 7. Build model
# model = Sequential()
# model.add(Dense(128, input_dim=10, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# # 8. Compile and train
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(
#     monitor='val_loss',
#     mode='min',
#     patience=50,
#     restore_best_weights=True
# )

# model.fit(
#     x_train, y_train,
#     epochs=100000,
#     batch_size=64,
#     validation_split=0.2,
#     verbose=1,
#     callbacks=[es]
# )

# # 9. Evaluate
# loss, acc = model.evaluate(x_test, y_test)
# print("accuracy :", acc)

# # 10. Predict on test_csv (scaled)
# y_submit = model.predict(test_scaled)
# y_submit = np.round(y_submit).astype(int)

# # 11. Save submission
# submission_csv['Exited'] = y_submit
# submission_csv.to_csv(path + 'submission_0527_scaled.csv')
# print("✅ Submission file saved: submission_0527_scaled.csv")



import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score

# 1. Load Data
path = 'Study25/_data/kaggle/bank/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. Encode categorical features
le_geo = LabelEncoder()
le_gender = LabelEncoder()

train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])

test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

# 3. Drop unneeded columns
train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# 4. Separate features and target
x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

# 5. Apply MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
test_scaled = scaler.transform(test_csv)

# 6. Train-test split (after scaling)
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, train_size=0.8, random_state=33
)

# 7. Build model
model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 8. Compile and train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

model.fit(
    x_train, y_train,
    epochs=100000,
    batch_size=64,
    validation_split=0.2,
    verbose=1,
    callbacks=[es]
)

# 9. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("accuracy :", acc)

# 10. Predict and round
y_submit = model.predict(test_scaled)
y_submit = np.round(y_submit).astype(int)

# 11. Save to CSV
submission_csv['Exited'] = y_submit
submission_csv.to_csv(path + 'submission_0527_minmax.csv')
print("✅ Submission file saved: submission_0527_minmax.csv")

