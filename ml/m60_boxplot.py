
from sklearn.datasets import fetch_california_housing

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import pandas as pd
import random
import matplotlib.pyplot as plt
seed = 333
random.seed(seed)
np.random.seed(seed)

from sklearn.ensemble import BaggingRegressor, BaggingClassifier, VotingRegressor, VotingClassifier



dataset= fetch_california_housing()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print(df)

df['target']  = dataset.target


df.boxplot()
plt.show()