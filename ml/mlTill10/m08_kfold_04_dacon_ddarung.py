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


path = './Study25/_data/dacon/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
# print(train_csv) #[1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
# print(test_csv) #[715 rows x 9 columns]

submission_csv = pd.read_csv(path+ 'submission.csv', index_col=0)
# print(submission_csv) # [715 rows x 1 columns]

train_csv = train_csv.dropna()


# test_csv = test_csv.dropna()
test_csv = test_csv.fillna(train_csv.mean())
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)  #count라는 axis=1 열 삭제, 행은 axis =0
print(x) #[1459 rows x 9 columns]

y = train_csv['count'] 
print(y) #(1459,)

n_split = 5

kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)
#회귀는 stratfied 안된다
# kfold = StratifiedKFold(n_splits=n_split, shuffle= True, random_state=42)


# model
# model = RandomForestRegressor()
model = HistGradientBoostingRegressor()

# train
score = cross_val_score(model, x,y, cv=kfold)   #fit 포함됨

print('accuracy:', score, '\n평균 acc:',round( np.mean(score), 4) )



# accuracy: [0.78184218 0.8419361  0.79432104 0.71133372 0.74879392] 
# 평균 acc: 0.7756

# accuracy: [0.7967942  0.84412743 0.80405959 0.7304369  0.73106867] 
# 평균 acc: 0.7813