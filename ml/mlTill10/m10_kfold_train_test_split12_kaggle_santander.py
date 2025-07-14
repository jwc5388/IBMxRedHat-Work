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


# 1. 데이터 로드
path = './Study25/_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv')
test_csv = pd.read_csv(path + 'test.csv')
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 2. 피처/타겟 분리 및 기본 피처 리스트 생성
y = train_csv['target']
x = train_csv.drop(['ID_code', 'target'], axis=1)
x_submit = test_csv.drop(['ID_code'], axis=1)
original_features = [col for col in x.columns]


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

# acc: [0.8576875  0.8606875  0.8636875  0.8589375  0.85921875] 
# 평균 acc: 0.86
# cross_val_predict ACC 0.880875