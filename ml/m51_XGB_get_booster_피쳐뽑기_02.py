
from sklearn.datasets import load_diabetes, load_breast_cancer, fetch_california_housing
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

datasets = fetch_california_housing()
x = datasets.data
y= datasets.target
print(x.shape, y.shape)         

feature_names = datasets.feature_names


# (20640, 8) (20640,)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(feature_names)
# exit()

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed)
                                                    # stratify=y)

es = xgb.callback.EarlyStopping(
    rounds = 50, 
    # metric_name = 'mlogloss',
    data_name = 'validation_0',
    # save_best = True,
    
)

model = XGBRegressor(
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




print('r2:', model.score(x_test, y_test))   
score_dict = model.get_booster().get_score(importance_type='gain')

total = sum(score_dict.values())
print(total) #25.03644559904933

# score_list = [score_dict.get(f"f{i}", 0) / total for i in range(x.shape[1])]
score_list = [v / total for v in score_dict.values()]


print(score_list)
# exit()

#첫번째 f는 formatted string 으로 줄여서 f-string 이라도 부른다
#두번째 f는 그냥 문자 f다
#f{i} = f 더하기 '0=> f0

thresholds = np.sort(score_list)

####컬럼명 매칭 #########

# score_df = pd.DataFrame({
#     'feature' : [feature_names[int(f[1:])]    for f in score_dict.keys()],
#     'gain' : list(score_dict.values())
# }).sort_values(by='gain', ascending=True)

score_df = pd.DataFrame({
    'feature' : feature_names,
    'gain' : list(score_dict.values())
}).sort_values(by='gain', ascending=True)

print(score_df)



from sklearn.feature_selection import SelectFromModel

delete_columns = []
max_r2 = 0
for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=True)  # model은 학습된 상태
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    mask = selection.get_support()
    
    # print('선택된 피쳐:', mask)
    not_select_features = [feature_names[j] 
           for j, selected in enumerate(mask) 
           if not selected]
    
    # print('삭제할 컬럼:', not_select_features)
        
        
    
    
    
    if select_x_train.shape[1] == 0:
        print(f"⚠️ Skipped: No features selected at threshold {i:.4f}")
        continue

    print(f"Threshold: {i:.4f}, n_features: {select_x_train.shape[1]}")
    
    select_model = XGBRegressor(
        n_estimators = 500,
        random_state = seed,
        gamma = 0,
        min_child_weight = 0,
        reg_alpha = 0,
        reg_lambda = 1,
        # eval_metric = 'mlogloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
        #                         # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
        # callbacks = [es]
         
        )
    
    select_model.fit(select_x_train, y_train, eval_set =[(select_x_test,y_test)], verbose=0)
    
    select_y_pred = select_model.predict(select_x_test)
    score = r2_score(y_test, select_y_pred)
    print('Trech=%.3f, n=%d, acc: %.4f%%' %(i, select_x_train.shape[1], score*100))
    
    
    
    # # 삭제된 컬럼명 구하기
    # removed_features = [feature_names[i] for i, selected in enumerate(selection.get_support()) if not selected]

    print('삭제할 컬럼:', not_select_features)
    print('=================================================')
    #delete columns =[]
    
    if score>= max_r2:
        delete_columns = not_select_features
        max_r2 = score
        nn = select_x_train.shape[1]
        
print('for max score deleted columns:', delete_columns)
print('max r2 score:', max_r2)
print('남은 컬럼수:', nn, '개')

# =================================================
# for max score deleted columns: ['AveBedrms', 'Population']
# max r2 score: 0.8410417274220061
# 남은 컬럼수: 6 개
    