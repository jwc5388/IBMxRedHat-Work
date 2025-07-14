
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
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

# 5. Apply MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
test_scaled = scaler.transform(test_csv)

# 6. Train-test split (after scaling)
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, train_size=0.8, random_state=33
)

path_mcp = 'Study25/_save/keras28_mcp/08_kaggle_bank/'
model = load_model(path_mcp + 'keras28_kaggle_bank_save.h5')

# 8. Compile and train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

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
