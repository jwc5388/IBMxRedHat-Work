from sklearn.datasets import load_iris, fetch_covtype
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import random
import numpy as np
import matplotlib.pyplot as plt
#1data

seed = 123

random.seed(seed)
np.random.seed(seed)

datasets = fetch_covtype()
x = datasets.data
y= datasets.target
print(x.shape, y.shape)    


# ✅ LabelEncoder 적용
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)



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
plot_importance(model)

plt.show()



    