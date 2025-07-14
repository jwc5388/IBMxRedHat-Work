import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

from sklearn.utils import all_estimators
import sklearn as sk
print(sk.__version__)  # 1.6.1

x,y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=42, train_size=0.8)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



# model = RandomForestRegressor()
allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgoriths: ', allAlgorithms)


print('모델의 갯수:', len(allAlgorithms))   #55
print(type(allAlgorithms))  #<class 'list'>

max_score =0
max_name = []
for (name, algorithm) in allAlgorithms: 
    try:
        model = algorithm()
        #train
        model.fit(x_train, y_train)
        
        result = model.score(x_test, y_test)
        
        if(result>max_score):
            max_score = result
            max_name = name
        else:
            max_score = max_score
        
        print(name, '의 정답률 : ', result)
        
    except:
        print(name, '은(는) 에러뜬 부분!!!' )
        
print('======================================================')
print('최고모델:', max_name, max_score)
print('======================================================')

# 최고모델: HistGradientBoostingRegressor 0.8364689118581992
