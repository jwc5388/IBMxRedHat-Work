
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler, MaxAbsScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint


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


# 6. Train-test split (after scaling)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=33
)

# 5. Apply Scalers
# scaler = MinMaxScaler()
# loss: 0.326958030462265
# accuracy : 0.8600296974182129
# r2 score: 0.3996829390525818

# scaler = StandardScaler()
# loss: 0.32714515924453735
# accuracy : 0.8622413277626038
# r2 score: 0.401749849319458

# scaler = RobustScaler()
# loss: 0.3283367455005646
# accuracy : 0.8621504306793213
# r2 score: 0.399594783782959

scaler = MaxAbsScaler()
# loss: 0.3268687129020691
# accuracy : 0.8614839315414429
# r2 score: 0.4008539915084839

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


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
    epochs=1000,
    batch_size=64,
    validation_split=0.2,
    verbose=1,
    callbacks=[es]
)

# 9. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("loss:", loss)
print("accuracy :", acc)

result = model.predict(x_test)
r2 =r2_score(y_test, result)
print("r2 score:", r2)

# 10. Predict and round
y_submit = model.predict(test_csv)
y_submit = np.round(y_submit).astype(int)

# 11. Save to CSV
# submission_csv['Exited'] = y_submit
# submission_csv.to_csv(path + 'submission_0527_minmax.csv')
# print("✅ Submission file saved: submission_0527_minmax.csv")
