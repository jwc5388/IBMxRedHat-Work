import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC

import pandas as pd

#1 data
dataset = load_iris()
x = dataset.data
y = dataset.target
# y = dataset['target']

print(x)
print(y)

df = pd.DataFrame(x, columns=dataset.feature_names)

print(df)


n_split = 3

kfold = KFold(n_splits= n_split, shuffle=False)


# for train_index, val_index in kfold.split(df):
#     print('======================================================================')
#     print(train_index, '\n',  val_index)
    
    
    


for i, (train_index, val_index) in enumerate(kfold.split(df)):
    print(f'=======================', [i], '===========================')
    print(train_index, '\n', val_index)
    
    
    
    