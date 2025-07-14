from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.datasets import fetch_covtype,load_digits,load_wine, load_iris, load_breast_cancer

import numpy as np
import pandas as pd
import time


#1 data
data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_digits(return_X_y=True),
             load_wine(return_X_y=True),
             ]

model_list = [LinearSVC(), 
              LogisticRegression(),
              DecisionTreeClassifier(),
              RandomForestClassifier()]



name_list = ['Iris', 'Breast Cancer', 'Digits', 'Wine']

for data, (x,y) in enumerate(data_list):
    print(f"\n🧪 Dataset: {name_list[data]}")
    # Split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=42, stratify=y
    )
    for i, model in enumerate(model_list):
        model.fit(x_train, y_train)
        result = model.score(x_test, y_test)
        
        print(f"[{i}] {model.__class__.__name__}")
        print(f"    ✅ Validation Accuracy: {result:.4f}")





# 🧪 Dataset: Iris
# [0] LinearSVC
#     ✅ Validation Accuracy: 0.9667
# [1] LogisticRegression
#     ✅ Validation Accuracy: 0.9667
# [2] DecisionTreeClassifier
#     ✅ Validation Accuracy: 0.9667
# [3] RandomForestClassifier
#     ✅ Validation Accuracy: 0.9667

# 🧪 Dataset: Breast Cancer
# [0] LinearSVC
#     ✅ Validation Accuracy: 0.9737
# [1] LogisticRegression
#     ✅ Validation Accuracy: 0.9561
# [2] DecisionTreeClassifier
#     ✅ Validation Accuracy: 0.9123
# [3] RandomForestClassifier
#     ✅ Validation Accuracy: 0.9561

# 🧪 Dataset: Digits
# [0] LinearSVC
#     ✅ Validation Accuracy: 0.9444
# [1] LogisticRegression
#     ✅ Validation Accuracy: 0.9583
# [2] DecisionTreeClassifier
#     ✅ Validation Accuracy: 0.8278
# [3] RandomForestClassifier
#     ✅ Validation Accuracy: 0.9667

# 🧪 Dataset: Wine
# [0] LinearSVC
#     ✅ Validation Accuracy: 0.9167
# [1] LogisticRegression
#     ✅ Validation Accuracy: 0.9722
# [2] DecisionTreeClassifier
#     ✅ Validation Accuracy: 0.9722
# [3] RandomForestClassifier
#     ✅ Validation Accuracy: 1.0000
# \