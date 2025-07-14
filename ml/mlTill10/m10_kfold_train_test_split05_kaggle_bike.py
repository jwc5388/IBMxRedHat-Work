
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

import pandas as pd


path = './Study25/_data/kaggle/bike/'
path_save = './Study25/_data/kaggle/bike/csv_files/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path+ 'sampleSubmission.csv', index_col=0)


# Prepare features and target
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1)
y = train_csv['count']

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



# acc: [0.33936975 0.34108406 0.32575495 0.36437109 0.35732977] 
# # 평균 acc: 0.3456

# r2 score: 0.27759641780213407