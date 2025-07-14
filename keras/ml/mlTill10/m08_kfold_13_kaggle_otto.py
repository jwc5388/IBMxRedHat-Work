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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import pandas as pd
from sklearn.utils import all_estimators
from sklearn.datasets import load_wine
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import HistGradientBoostingClassifier




# 1. Load Data
path = './Study25/_data/kaggle/otto/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# 2. Feature & Target 분리
x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

n_split = 5

# kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)


#회귀는 stratfied 안된다
kfold = StratifiedKFold(n_splits=n_split, shuffle= True, random_state=42)


# model
# model = RandomForestClassifier()
model = HistGradientBoostingClassifier()


# train
score = cross_val_score(model, x,y, cv=kfold)   #fit 포함됨

print('accuracy:', score, '\n평균 acc:',round( np.mean(score), 4) )


#       accuracy: [0.8072075  0.8071267  0.81738849 0.80945455 0.81034343] 
# 평균 acc: 0.8103