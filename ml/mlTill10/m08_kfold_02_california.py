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

from sklearn.utils import all_estimators
import sklearn as sk
print(sk.__version__)  # 1.6.1


#1데이터
x,y = fetch_california_housing(return_X_y=True)

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


#random
#  accuracy: [0.80647079 0.81516254 0.80113099 0.82598943 0.80546025] 
# 평균 acc: 0.8108


#histgrad

# accuracy: [0.83642746 0.842161   0.82748581 0.84793045 0.82897002] 
# 평균 acc: 0.8366

