import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

# 1. Load Data
path = './Study25/_data/kaggle/otto/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# 2. Feature & Target 분리
x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42, stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5

kfold = StratifiedKFold(n_splits= n_split, shuffle=True, random_state=42)

#2 model
model = MLPClassifier()


#3 훈련 
score = cross_val_score(model, x_train, y_train, cv = kfold)

print('acc:', score, '\n평균 acc:', round(np.mean(score), 4) )


y_pred = cross_val_predict(model, x_test, y_test, cv = kfold)

acc_score = accuracy_score(y_test, y_pred)

print('cross_val_predict ACC', acc_score)



# acc: [0.77547722 0.78113322 0.78444444 0.78636364 0.78282828] 
# 평균 acc: 0.782
# cross_val_predict ACC 0.7520200387847447