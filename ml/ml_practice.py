# # # # # # from keras.models import Sequential, load_model
# # # # # # from keras.layers import Dense, Conv1D, Conv2D, Flatten, BatchNormalization
# # # # # # from sklearn.model_selection import train_test_split,  KFold, StratifiedKFold
# # # # # # from sklearn.datasets import load_diabetes
# # # # # # from sklearn.metrics import r2_score, mean_squared_error
# # # # # # from keras.callbacks import EarlyStopping, ModelCheckpoint

# # # # # # from sklearn.preprocessing import StandardScaler, MinMaxScaler
# # # # # # import numpy as np
# # # # # # import time
# # # # # # from sklearn.svm import LinearSVC
# # # # # # from sklearn.linear_model import LogisticRegression
# # # # # # from sklearn.tree import DecisionTreeClassifier
# # # # # # from sklearn.ensemble import RandomForestClassifier
# # # # # # import pandas as pd
# # # # # # from sklearn.model_selection import train_test_split
# # # # # # from sklearn.datasets import load_wine
# # # # # # from keras.utils import to_categorical
# # # # # # from sklearn.datasets import fetch_covtype



# # # # # # model_list = [LinearSVC(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]


# # # # # # x, y = fetch_covtype(return_X_y=True)

# # # # # # x_train, x_test, y_train, y_test= train_test_split(x,y, train_size=0.2, random_state=42)

# # # # # # for model in model_list:
# # # # # #     model.fit(x_train, y_train)
# # # # # #     result = model.score(x_test, y_test)
# # # # # #     print(f" {type(model).__name__} Validation Accuracy: {result : .4f}")
    
    
# # # # # from sklearn.datasets import fetch_california_housing
# # # # # from sklearn.model_selection import train_test_split
# # # # # from sklearn.preprocessing import RobustScaler
# # # # # from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
# # # # # from sklearn.metrics import r2_score
# # # # # import numpy as np
# # # # # import warnings
# # # # # warnings.filterwarnings('ignore')

# # # # # from sklearn.utils import all_estimators
# # # # # import sklearn as sk

# # # # # x , y = fetch_california_housing(return_X_y=True)

# # # # # kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# # # # # scaler = RobustScaler()
# # # # # x = scaler.fit_transform(x)

# # # # # allAlgorithms = all_estimators(type_filter='regressor')



# # # # # max_score = 0
# # # # # max_name = None
# # # # # model_scores = []
# # # # # for (name, algorithm) in allAlgorithms:
# # # # #     try:
# # # # #         model = algorithm()
# # # # #         scores =cross_val_score(model, x, y, cv = kfold, scoring='r2')
# # # # #         mean_score = np.mean(scores)
# # # # #         model_scores.append((name, mean_score))
        
        
# # # # #         print()


# # # # from sklearn.datasets import fetch_california_housing
# # # # from sklearn.tree import DecisionTreeClassifier
# # # # from sklearn.ensemble import RandomForestClassifier
# # # # from sklearn.model_selection import train_test_split
# # # # from sklearn.ensemble import GradientBoostingClassifier
# # # # from xgboost import XGBClassifier, XGBRegressor
# # # # import random
# # # # import numpy as np
# # # # import pandas as pd
# # # # from sklearn.metrics import accuracy_score, r2_score
# # # # import matplotlib.pyplot as plt
# # # # import xgboost as xgb

# # # # seed = 42
# # # # random.seed(seed)
# # # # np.random.seed(seed)

# # # # dataset = fetch_california_housing()
# # # # x = dataset.data
# # # # y = dataset['target']

# # # # feature_names = dataset.feature_names

# # # # print(feature_names)

# # # # # print(y)

# # # # x_train , x_test, y_train, y_test= train_test_split(x,y, random_state=42, train_size=0.8)

# # # # es = xgb.callback.EarlyStopping(
# # # #     rounds = 20,
# # # #     data_name = 'validation_0'
# # # # )

# # # # model = XGBRegressor(
# # # #     n_estimators = 300,
# # # #     random_state = seed,
# # # #     gamma = 0,
# # # #     min_child_weight = 0,
# # # #     reg_alpha = 0,
# # # #     reg_lambda = 1,
# # # #     callbacks = [es]
# # # # )


# # # # model.fit(x_train, y_train, eval_set =[(x_test, y_test)], verbose  =1)

# # # # print('r2:', model.score(x_test, y_test))
# # # # score_dict = model.get_booster().get_score(importance_type = 'gain')

# # # # total = sum(score_dict.values())
# # # # print(total)

# # # # score_list = [v/total for v in score_dict.values()]


# # # # thresholds= np.sort(score_list)

# # # # score_df = pd.DataFrame({
# # # #     'feature' : feature_names,
# # # #     'gain' : list(score_dict.values()),
    
# # # # }).sort_values(by='gain', ascending=True)

# # # # print(score_df)

# # # # from sklearn.feature_selection import SelectFromModel

# # # # delete_columns = []
# # # # max_r2 = 0

# # # # for i in thresholds:
# # # #     selection = SelectFromModel(model, threshold=i, prefit=True)
# # # #     select_x_train = selection.transform(x_train)
# # # #     select_x_test = selection.transform(x_test)
    
# # # #     mask = selection.get_support()
    
    
# # # #     not_select_features = [feature_names[j]
# # # #                            for j, selected in enumerate(mask)
# # # #                            if not selected ]
    
    
# # # #     if select_x_train.shape[1] == 0:
# # # #         print(f"⚠️ Skipped: No features selected at threshold {i:.4f}")
# # # #         continue

# # # #     print(f"Threshold: {i:.4f}, n_features: {select_x_train.shape[1]}")
    
    
# # # #     select_model = XGBRegressor(
# # # #         n_estimators = 300,
# # # #         random_state = seed,
# # # #         gamma = 0,
# # # #         min_child_weight = 0,
# # # #         reg_alpha = 0,
# # # #         reg_lambda = 1,
        
# # # #     )
    
# # # #     select_model.fit(select_x_test, y_train, eval_set = [(select_x_test, y_test)])
    
    
# # # #     select_y_pred = model.predict(select_x_test)
# # # #     score = r2_score(y_test, select_y_pred)
    
# # # #     print('Trech=%.3f, n=%d, acc: %.4f%%' %(i, select_x_train.shape[1], score*100))

# # # #     print('columns to delete:::', not_select_features)
# # # #     print('===================================================')
    
# # # #     if score>= max_r2:
# # # #         delete_columns = not_select_features
# # # #         max_r2 = score
# # # #         nn = select_x_train.shape[1]
        
        
# # # # print('for max score deleted columns:', delete_columns)
# # # # print('max r2 score:', max_r2)
# # # # print('남은 컬럼수:', nn, '개')    


# # # from sklearn.datasets import load_iris
# # # from sklearn.tree import DecisionTreeClassifier
# # # from sklearn.ensemble import RandomForestClassifier
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.ensemble import GradientBoostingClassifier
# # # from xgboost import XGBClassifier
# # # import random
# # # import numpy as np
# # # import xgboost as xgb
# # # import pandas as pd
# # # from sklearn.metrics import accuracy_score, r2_score


# # # seed = 42ㅌ

# # # random.seed(seed)
# # # np.random.seed(seed)

# # # dataset = load_iris()
# # # x = dataset.data
# # # y = dataset['target']

# # # x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=42)

# # # es = xgb.callback.EarlyStopping(
# # #     rounds = 50,
# # #     metric_name = 'mlogloss',
# # #     data_name = 'validation_0',
# # # )

# # # model = XGBClassifier(
# # #     n_estimators = 500,
# # #     random_state = seed,
# # #     gamma = 0,
# # #     min_child_weight = 0,
# # #     reg_alpha = 0,
# # #     reg_lambda = 1,
# # #     eval_metric = 'mlogloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
# # #                             # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
# # #     callbacks = [es]
    
# # #     )

# # # model.fit(x_train, y_train, eval_set = [(x_test, y_test)], verbose=0)

# # # print('acc:', model.score(x_test, y_test))
# # # print(model.feature_importances_)


# # # thresholds = np.sort(model.feature_importances_)
# # # print(thresholds)

# # # from sklearn.feature_selection import SelectFromModel

# # # for i in thresholds:
# # #     selection = SelectFromModel(model, threshold=i, prefit=False)
# # #     select_x_train = selection.transform(x_train)
# # #     select_x_test = selection.transform(x_test)
    
# # #     select_model = XGBClassifier(
# # #         n_estimators = 500,
# # #         random_state = seed,
# # #         gamma = 0,
# # #         min_child_weight = 0,
# # #         reg_alpha = 0,
# # #         reg_lambda = 1,
# # #         eval_metric = 'mlogloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
# # #                                 # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
# # #         callbacks = [es]
        
# # #         )
    
# # #     select_model.fit(select_x_train, y_train, eval_set = [(select_x_test, y_test)], verbose = 0)
    
# # #     select_y_pred = select_model.predict(select_x_test)
# # #     score = accuracy_score(select_y_pred, y_test)
# # #     print('Trech=%.3f, n=%d, ACC: %.4f%%' %(i, select_x_train.shape[1], score*100))


# # from sklearn.datasets import load_iris
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import GradientBoostingClassifier
# # from xgboost import XGBClassifier
# # import random
# # import numpy as np
# # import xgboost as xgb
# # import pandas as pd
# # from sklearn.metrics import accuracy_score

# # seed = 42
# # random.seed(seed)
# # np.random.seed(seed)

# # dataset = load_iris()
# # x = dataset.data
# # y = dataset['target']

# # x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=42)


# # es = xgb.callback.EarlyStopping(
# #     rounds = 50,
# #     metric_name = 'mlogloss',
# #     data_name = 'validation_0',
# # )

# # model = XGBClassifier(
# #     n_estimators = 500,
# #     random_state = seed,
# #     gamma = 0,
# #     min_child_weight = 0,
# #     reg_alpha = 0,
# #     reg_lambda = 1,
# #     eval_metric = 'mlogloss',
# #     callbacks = [es]
# # )

# # model.fit(x_train, y_train, eval_set = [(x_test, y_test)], verbose = 0)

# # print('acc:', model.score(x_test, y_test))
# # print(model.feature_importances_)

# # thresholds = np.sort(model.feature_importances_)
# # print(thresholds)

# # from sklearn.feature_selection import SelectFromModel

# # for i in thresholds:
# #     selection = SelectFromModel(model, threshold=i, prefit = False)
# #     select_x_train = selection.transform(x_train)
# #     select_x_test = selection.transform(x_test)
    
# #     select_model = XGBClassifier(
# #         n_estimators = 500,
# #         random_state = seed,
# #         gamma = 0,
# #         min_child_weight = 0,
# #         reg_alpha = 0,
# #         reg_lambda = 1, 
# #         eval_metric = 'mlogloss',
# #         callbacks = [es]
        
        
# #     )
    
# #     select_model.fit(select_x_train, y_train, eval_set = [(select_x_test, y_test)], verbose = 0)
# import random
# import numpy as np
# from sklearn.datasets import load_iris

# seed = 42    
# random.seed(seed)
# np.random.seed(seed)

# dataset = load_iris()

import numpy as np
import pandas as pd
import random

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


dataset = fetch_covtype()
x = dataset.data
y = dataset['target']

print(x.shape, y.shape)
print(np.unique(y, return_counts= True))

x_train, x_test,  y_train, y_test= train_test_split(x,y, random_state=seed, train_size=0.8, shuffle=True, stratify=y)

from imblearn.over_sampling import SMOTE
import sklearn as sk

smote = SMOTE(random_state=seed,
              k_neighbors=5,
              sampling_strategy='auto')


x_train, y_train = smote.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts= True))

model = Sequential()
model.add(Dense(10, input_shape = (x.shape[1],0)))
model.add(Dense(7, activation='softmax'))

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc'])

from keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor = 'val_loss',
    mode= 'auto',
    restore_best_weights=True,
    patience = 10
)

model.fit(x_train, y_train, epochs = 30, validation_split=0.2, batch_size=32)


result = model.evaluate(x_test, y_test)
print('')