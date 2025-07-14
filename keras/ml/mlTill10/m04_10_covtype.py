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
from sklearn.datasets import load_wine
from keras.utils import to_categorical
from sklearn.datasets import fetch_covtype



model_list = [LinearSVC(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]

# 1. Load data
datasets = fetch_covtype()
x = datasets.data
y = datasets.target  # labels: 1 ~ 7
# 3. Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42
)


# for i, model in enumerate(model_list):
#     model.fit(x_train, y_train)
#     result = model.score(x_test, y_test)
    
#     print(f"[{i}] {model.__class__.__name__}")
#     print(f"    ✅ Validation Accuracy: {result:.4f}")
    
    
for model in model_list:
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    
    print(f"✅ {type(model).__name__} Validation Accuracy: {result:.4f}")

# [0] LinearSVC
#     ✅ Validation Accuracy: 0.7047
# [1] LogisticRegression
#     ✅ Validation Accuracy: 0.6223
# [2] DecisionTreeClassifier
#     ✅ Validation Accuracy: 0.9389
# [3] RandomForestClassifier
#     ✅ Validation Accuracy: 0.9556