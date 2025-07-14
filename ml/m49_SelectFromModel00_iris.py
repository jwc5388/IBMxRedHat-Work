from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import random
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score
#1data

seed = 42

random.seed(seed)
np.random.seed(seed)

datasets = load_iris()
x = datasets.data
y= datasets.target
print(x.shape, y.shape)         #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed, stratify=y)

es = xgb.callback.EarlyStopping(
    rounds = 50, 
    metric_name = 'mlogloss',
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
    eval_metric = 'mlogloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
                            # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
    callbacks = [es]
    
    )


model.fit(x_train, y_train, eval_set =[(x_test,y_test)], verbose=0)




print('acc2:', model.score(x_test, y_test))   # acc2: 0.9333333333333333
print(model.feature_importances_)# [0.01230742 0.02487084 0.5794107  0.38341108]


thresholds = np.sort(model.feature_importances_) #오름차순
print(thresholds) #[0.01230742 0.02487084 0.38341108 0.5794107 ]

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit = False)
    #threshold 가 i값 이상인것을 모두 훈련시킨다. 
    #prefit = False: 모델이 아직 학습되지 않았을떄, fit호출해서 훈련한다(기본값)
    #prefit = True: 이미 학습된 모델을 전달할때, fit 호출하지 않음.
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(select_x_train.shape) 

    select_model = XGBClassifier(
        n_estimators = 500,
        random_state = seed,
        gamma = 0,
        min_child_weight = 0,
        reg_alpha = 0,
        reg_lambda = 1,
        eval_metric = 'mlogloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
                                # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
        callbacks = [es]
        
        )
    
    select_model.fit(select_x_train, y_train, eval_set =[(select_x_test,y_test)], verbose=0)
    
    select_y_pred = select_model.predict(select_x_test)
    score = accuracy_score(select_y_pred, y_test)
    print('Trech=%.3f, n=%d, ACC: %.4f%%' %(i, select_x_train.shape[1], score*100))
    # print(score)
    # print('acc2:', model.score(select_x_test, y_test))  
        








# print("===============", model.__class__.__name__, "======================")
# print('acc:', model.score(x_test, y_test))      #acc: 0.9333333333333333
# print(model.feature_importances_)


# print('25%지점:', np.percentile(model.feature_importances_, 25))   #0.024616712238639593


# # [0.02430454 0.02472077 0.7376847  0.21328996]
# percentile = np.percentile(model.feature_importances_, 25)
# print(type(percentile))
# # for i, fi in enumerate(model.fe)
# print(xgb.__version__) #내껀 1.7.6 다른 2.1.4


# col_name = []
# #삭제할 컬럼(25% 이하인 놈들) 을 찾아내자!!
# for i, fi in enumerate(model.feature_importances_):
#     # print(i, fi)
#     if fi <= percentile:
#         col_name.append(datasets.feature_names[i])
#     else:
#         continue
    
# print(col_name) #['sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# x = pd.DataFrame(x, columns = datasets.feature_names)
# x = x.drop(columns = col_name)

# # print(x)


# x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed, stratify=y)

# model.fit(x_train, y_train)