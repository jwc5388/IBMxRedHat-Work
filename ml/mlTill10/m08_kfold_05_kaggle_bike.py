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


path = './Study25/_data/kaggle/bike/'
path_save = './Study25/_data/kaggle/bike/csv_files/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path+ 'sampleSubmission.csv', index_col=0)


# Prepare features and target
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1)
y = train_csv['count']

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


#       accuracy: [0.29392008 0.26624445 0.30229269 0.27587305 0.32161134] 
# 평균 acc: 0.292


#       accuracy: [0.37270208 0.34557704 0.3664712  0.32788793 0.36340218] 
# 평균 acc: 0.3552