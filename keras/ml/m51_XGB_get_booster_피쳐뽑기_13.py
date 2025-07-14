
from sklearn.datasets import load_diabetes, load_breast_cancer, load_wine, fetch_covtype, load_digits
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
from sklearn.preprocessing import StandardScaler

#1data

seed = 42

random.seed(seed)
np.random.seed(seed)

import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

# 1. Load Data
path = basepath + '_data/kaggle/otto/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# 2. Feature & Target 분리
x = train_csv.drop(['target'], axis=1)
feature_names = x.columns
y = train_csv['target']



# # ✅ LabelEncoder 적용
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42, stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


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
    'feature': [feature_names[int(f[1:])] for f in score_dict.keys()],
    'gain': list(score_dict.values())
}).sort_values(by='gain', ascending=True)

# score_df = pd.DataFrame({
#     'feature' : feature_names,
#     'gain' : list(score_dict.values())
# }).sort_values(by='gain', ascending=True)

print(score_df)



from sklearn.feature_selection import SelectFromModel

delete_columns = []
max_acc = 0
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
    
    select_model = XGBClassifier(
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
    score = accuracy_score(y_test, select_y_pred)
    print('Trech=%.3f, n=%d, acc: %.4f%%' %(i, select_x_train.shape[1], score*100))
    
    
    
    # # 삭제된 컬럼명 구하기
    # removed_features = [feature_names[i] for i, selected in enumerate(selection.get_support()) if not selected]

    print('삭제할 컬럼:', not_select_features)
    print('=================================================')
    #delete columns =[]
    
    if score>= max_acc:
        delete_columns = not_select_features
        max_acc = score
        nn = select_x_train.shape[1]
        
print('for max score deleted columns:', delete_columns)
print('max accuracy score:', max_acc)
print('남은 컬럼수:', nn, '개')

# =================================================


#     for max score deleted columns: ['feat_2', 'feat_6', 'feat_12']
# max accuracy score: 0.8283775048480931
# 남은 컬럼수: 90 개