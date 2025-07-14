
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

# Load Data
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

x = data
y = target
print(x.shape, y.shape)  # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5

kfold = KFold(n_splits= n_split, shuffle=True, random_state=42)

#2 model
model = HistGradientBoostingRegressor()


#3 훈련 
score = cross_val_score(model, x_train, y_train, cv = kfold)

print('acc:', score, '\n평균 acc:', round(np.mean(score), 4) )



y_pred = cross_val_predict(model, x_test, y_test, cv = kfold)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print('r2 score:', r2)

# acc_score = accuracy_score(y_test, y_pred)

# print('cross_val_predict ACC', acc_score)

# 평균 acc: 0.8337

# r2 score: 0.5697648097415464