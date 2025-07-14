from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine, load_digits, fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler
import random
import numpy as np

#1data

seed = 33

random.seed(seed)
np.random.seed(seed)

data1 = fetch_california_housing()
data2 = load_diabetes()

datasets = [data1, data2]
dataset_name = ['california', 'diabetes']

model1 = DecisionTreeRegressor(random_state=seed)
model2 = RandomForestRegressor(random_state=seed)
model3 = GradientBoostingRegressor(random_state = seed)
model4 = XGBRegressor(random_state = seed)
models = [model1, model2, model3, model4]

for i,data in enumerate(datasets):
    x = data.data
    y = data.target
    

    x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    print("==========================================", dataset_name[i], "==============================================" )
    
    #2 model 구성
    for model in models:
        model.fit(x_train, y_train)
        print("===============", model.__class__.__name__, "======================")
        print('r2:', model.score(x_test, y_test))
        print(model.feature_importances_)
        
    
# ========================================== california ==============================================
# =============== DecisionTreeRegressor ======================
# r2: 0.6196996034347311
# [0.51409842 0.05382381 0.05427861 0.02813063 0.03449004 0.13728032
#  0.08800113 0.08989705]
# =============== RandomForestRegressor ======================
# r2: 0.8232000302401453
# [0.51622773 0.05351559 0.05041949 0.03136796 0.03299399 0.14099852
#  0.08717871 0.087298  ]
# =============== GradientBoostingRegressor ======================
# r2: 0.8046028102852306
# [0.59337311 0.02797374 0.01974463 0.00518707 0.00381886 0.13192919
#  0.09610003 0.12187337]
# =============== XGBRegressor ======================
# r2: 0.8460473198002185
# [0.43224457 0.0746105  0.04970725 0.0278855  0.02821121 0.15559179
#  0.11284696 0.11890225]
# ========================================== diabetes ==============================================
# =============== DecisionTreeRegressor ======================
# r2: 0.0001320392733663578
# [0.05914146 0.00915029 0.25103079 0.07069226 0.05841692 0.09354296
#  0.0390023  0.01498835 0.32984763 0.07418705]
# =============== RandomForestRegressor ======================
# r2: 0.4934542553468001
# [0.05790437 0.01150833 0.30299422 0.09374478 0.04492863 0.0598607
#  0.06181206 0.01924143 0.27264336 0.07536213]
# =============== GradientBoostingRegressor ======================
# r2: 0.4596415743043656
# [0.06440215 0.01979657 0.26676194 0.07916043 0.02727053 0.07105739
#  0.05344689 0.01484591 0.36712225 0.03613594]
# =============== XGBRegressor ======================
# r2: 0.34316563388449806
# [0.03478482 0.061127   0.20532171 0.08024666 0.04313674 0.05784599
#  0.07921447 0.0511908  0.3334028  0.05372896]