from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
import matplotlib.pyplot as plt
#1data

seed = 123

random.seed(seed)
np.random.seed(seed)

datasets = fetch_california_housing()
x = datasets.data
y= datasets.target
print(x.shape, y.shape)         #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed)



#2 model
model1 = DecisionTreeRegressor(random_state=seed)
model2 = RandomForestRegressor(random_state=seed)
model3 = GradientBoostingRegressor(random_state = seed)
model4 = XGBRegressor(random_state = seed)

models = [model1, model2, model3, model4]
for model in models:
    model.fit(x_train, y_train)
    print("===============", model.__class__.__name__, "======================")
    print('r2:', model.score(x_test, y_test))
    print(model.feature_importances_)
    

from xgboost.plotting import plot_importance
plot_importance(model)

plt.show()



    