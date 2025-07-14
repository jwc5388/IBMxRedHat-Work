import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
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


n_split = 5

kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)
#회귀는 stratfied 안된다
# kfold = StratifiedKFold(n_splits=n_split, shuffle= True, random_state=42)


# model
model = RandomForestRegressor()
# model = HistGradientBoostingClassifier()

# train
score = cross_val_score(model, x,y, cv=kfold)   #fit 포함됨

print('accuracy:', score, '\n평균 acc:',round( np.mean(score), 4) )


# accuracy: [0.21558473 0.27347701 0.27714108 0.12096282 0.36120278] 
# 평균 acc: 0.2497


# accuracy: [ 0.08341454  0.05594328  0.26208709 -0.08174885  0.27680945] 
# 평균 acc: 0.1193


