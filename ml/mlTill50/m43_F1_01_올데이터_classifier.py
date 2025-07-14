from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine, load_digits
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

data1 = load_breast_cancer()
data2 = load_wine()
data3 = load_digits()

datasets = [data1, data2, data3]
dataset_name = ['cancer', 'wine', 'digits']

model1 = DecisionTreeClassifier(random_state=seed)
model2 = RandomForestClassifier(random_state=seed)
model3 = GradientBoostingClassifier(random_state = seed)
model4 = XGBClassifier(random_state = seed)
models = [model1, model2, model3, model4]

for i,data in enumerate(datasets):
    x = data.data
    y = data.target
    

    x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed, stratify=y)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    print("==========================================", dataset_name[i], "==============================================" )
    
    #2 model 구성
    for model in models:
        model.fit(x_train, y_train)
        print("===============", model.__class__.__name__, "======================")
        print('acc:', model.score(x_test, y_test))
        print(model.feature_importances_)
        
        
# ========================================== cancer ==============================================
# =============== DecisionTreeRegressor ======================
# r2: 0.5145758270553554

# =============== RandomForestRegressor ======================
# r2: 0.7887620700949886

# =============== GradientBoostingRegressor ======================
# r2: 0.7409926763917718

# =============== XGBRegressor ======================
# r2: 0.7010190355044268

# ========================================== wine ==============================================
# =============== DecisionTreeRegressor ======================
# r2: -0.23350253807106625

# =============== RandomForestRegressor ======================
# r2: 0.5963111675126903

# =============== GradientBoostingRegressor ======================
# r2: 0.7630100505594316

# =============== XGBRegressor ======================
# r2: 0.35176078145046485

# ========================================== digits ==============================================
# =============== DecisionTreeRegressor ======================
# r2: 0.6024949426837491

# =============== RandomForestRegressor ======================
# r2: 0.8819402562373567

# =============== GradientBoostingRegressor ======================
# r2: 0.813660809185701

# =============== XGBRegressor ======================
# r2: 0.848086143243896


   
















 ###Decisiontreeregressor seed 42
# r2: 0.060653981041140725
# [0.06458641 0.00667276 0.41823226 0.0624936  0.08317685 0.05338883
#  0.063936   0.0297882  0.15579517 0.06192992]
# =============== RandomForestRegressor ======================
# r2: 0.4428225673999313
# [0.05864167 0.00963304 0.35546898 0.08840759 0.05278353 0.05722749
#  0.05133862 0.02421276 0.23095698 0.07132935]
# =============== GradientBoostingRegressor ======================
# r2: 0.4529343796683364
# [0.04961219 0.0124781  0.39310125 0.08297929 0.03889154 0.06148979
#  0.0365877  0.02859617 0.24958299 0.04668098]
# =============== XGBRegressor ======================
# r2: 0.22857599305390852
# [0.02547606 0.06533135 0.28826523 0.06119404 0.05055214 0.06347557
#  0.04132882 0.09082817 0.25840732 0.05514139]


# seed 33
# =============== DecisionTreeRegressor ======================
# r2: 0.0001320392733663578
# [0.05914146 0.00915029 0.25103079 0.07069226 0.05841692 0.09354296
#  0.0390023  0.01498835 0.32984763 0.07418705]
# =============== RandomForestRegressor ======================
# r2: 0.49333536861462
# [0.05790437 0.01150833 0.30299422 0.09374478 0.04492863 0.0598607
#  0.06181206 0.01924143 0.27264336 0.07536213]
# =============== GradientBoostingRegressor ======================
# r2: 0.45996378449863373
# [0.06440215 0.01979657 0.26676194 0.07916043 0.02727053 0.07105739
#  0.05344689 0.01484591 0.36712225 0.03613594]
# =============== XGBRegressor ======================
# r2: 0.3310163419069171
# [0.03478482 0.061127   0.20532171 0.08024666 0.04313674 0.05784599
#  0.07921447 0.0511908  0.3334028  0.05372896]