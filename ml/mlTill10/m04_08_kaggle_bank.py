from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split


model_list = [LinearSVC(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]

# 1. Load Data
path = './Study25/_data/kaggle/bank/'

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


for i, model in enumerate(model_list):
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    
    print(f"[{i}] {model.__class__.__name__}")
    print(f"    ✅ Validation Accuracy: {result:.4f}")

# [0] LinearSVC
#     ✅ Validation Accuracy: 0.8187
# [1] LogisticRegression
#     ✅ Validation Accuracy: 0.8238
# [2] DecisionTreeClassifier
#     ✅ Validation Accuracy: 0.7947
# [3] RandomForestClassifier
#     ✅ Validation Accuracy: 0.8562