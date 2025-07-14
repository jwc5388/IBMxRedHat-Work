from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split


model_list = [LinearSVC(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]

#1 data
path = './Study25/_data/diabetes/'

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


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=42)



for i, model in enumerate(model_list):
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    
    print(f"[{i}] {model.__class__.__name__}")
    print(f"    ✅ Validation Accuracy: {result:.4f}")



# [0] LinearSVC
#     ✅ Validation Accuracy: 0.7939
# [1] LogisticRegression
#     ✅ Validation Accuracy: 0.8015
# [2] DecisionTreeClassifier
#     ✅ Validation Accuracy: 0.6947
# [3] RandomForestClassifier
#     ✅ Validation Accuracy: 0.7557