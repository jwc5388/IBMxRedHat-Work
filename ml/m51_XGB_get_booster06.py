
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt

#1data

seed = 42

random.seed(seed)
np.random.seed(seed)

datasets = load_breast_cancer()
x = datasets.data
y= datasets.target
print(x.shape, y.shape)         #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed,stratify=y)
                                                    # stratify=y)

es = xgb.callback.EarlyStopping(
    rounds = 50, 
    # metric_name = 'mlogloss',
    data_name = 'validation_0',
    # save_best = True,
    
)

model = XGBClassifier(
    n_estimators = 500,
    random_state = seed,
    gamma = 0,
    min_child_weight = 0,
    reg_alpha = 0,
    reg_lambda = 1,
    # eval_metric = 'mlogloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
    #                         # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
    callbacks = [es]
    
    )


model.fit(x_train, y_train, eval_set =[(x_test,y_test)], verbose=1)




print('acc:', model.score(x_test, y_test))   
# print(model.feature_importances_)
# aaa = model.get_booster().get_score(importance_type='weight') #split 빈도수 개념
# {'f0': 135.0, 'f1': 95.0, 'f2': 10.0, 'f3': 20.0, 'f4': 43.0, 'f5': 25.0, 'f6': 22.0, 'f7': 66.0, 'f8': 14.0, 'f9': 12.0, 
#  'f10': 47.0, 'f11': 40.0, 'f12': 9.0, 'f13': 52.0, 'f14': 25.0, 'f15': 21.0, 'f16': 14.0, 'f17': 10.0, 'f18': 11.0, 
#  'f19': 41.0, 'f20': 35.0, 'f21': 79.0, 'f22': 56.0, 'f23': 32.0, 'f24': 52.0, 'f25': 6.0, 'f26': 24.0, 'f27': 88.0, 
#  'f28': 70.0, 'f29': 3.0}
aaa = model.get_booster().get_score(importance_type='gain')

##GAIN~!@KSKJFKSDJ 을 믿어라~ over weight!!!

# print(aaa)




# exit()

gain_values = sorted(aaa.values())

# print(gain_values)

# exit()


from sklearn.feature_selection import SelectFromModel


for thresh in gain_values:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)  # model은 학습된 상태
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    if select_x_train.shape[1] == 0:
        print(f"⚠️ Skipped: No features selected at threshold {thresh:.4f}")
        continue

    print(f"Threshold: {thresh:.4f}, n_features: {select_x_train.shape[1]}")
    
    select_model = XGBClassifier(
        n_estimators = 500,
        random_state = seed,
        gamma = 0,
        min_child_weight = 0,
        reg_alpha = 0,
        reg_lambda = 1,
        # eval_metric = 'mlogloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
        #                         # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
        callbacks = [es]
         
        )
    
    select_model.fit(select_x_train, y_train, eval_set =[(select_x_test,y_test)], verbose=0)
    
    select_y_pred = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_pred)
    print('Trech=%.3f, n=%d, acc: %.4f%%' %(thresh, select_x_train.shape[1], score*100))
    
# from xgboost.plotting import plot_importance
# plot_importance(model)

# plt.show()


    