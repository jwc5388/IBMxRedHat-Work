from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import random
import numpy as np
import matplotlib.pyplot as plt
#1data

seed = 42

random.seed(seed)
np.random.seed(seed)

datasets = load_iris()
x = datasets.data
y= datasets.target
print(x.shape, y.shape)         #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed, stratify=y)



#2 model
model1 = DecisionTreeClassifier(random_state=seed)
model2 = RandomForestClassifier(random_state=seed)
model3 = GradientBoostingClassifier(random_state = seed)
model4 = XGBClassifier(random_state = seed)

models = [model1, model2, model3, model4]
for model in models:
    model.fit(x_train, y_train)
    print("===============", model.__class__.__name__, "======================")
    print('acc:', model.score(x_test, y_test))
    print(model.feature_importances_)
    

from xgboost.plotting import plot_importance
plot_importance(model, importance_type='gain', title= 'feature importance [gain]')

"""
weight : 얼마나 자주 split했냐, 통상 frequency
gain: split가 모델의 성능을 얼마나 개선했냐// 통상적으로 얘 많이 써
cover: split 하기 위한 sample 수. 별로. 
"""

#gain이 좀더 신뢰도 높음

plt.show()


    