import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd

from sklearn.utils import all_estimators
import sklearn as sk
print(sk.__version__)  # 1.6.1

# === Load Data ===
path = './Study25/_data/dacon/cancer/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# === Separate Features and Target Label ===
x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']


# === One-Hot Encode Categorical Columns ===
categorical_cols = x.select_dtypes(include='object').columns
x = pd.get_dummies(x, columns=categorical_cols)
test_csv = pd.get_dummies(test_csv, columns=categorical_cols)
x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)
n_split = 5

# kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)


#회귀는 stratfied 안된다

kfold = StratifiedKFold(n_splits=n_split, shuffle= True, random_state=42)


# model
# model = RandomForestClassifier()
model = MLPClassifier()

# train
score = cross_val_score(model, x,y, cv=kfold)   #fit 포함됨

print('accuracy:', score, '\n평균 acc:',round( np.mean(score), 4) )


#       accuracy: [0.88211335 0.88159706 0.88417852 0.88062184 0.88101658] 
# 평균 acc: 0.8819