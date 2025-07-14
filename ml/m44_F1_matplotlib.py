from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import random
import numpy as np

#1data

seed = 123

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
    
    
import matplotlib.pyplot as plt
def plot_feature_importance_datasets(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), model.feature_importances_)
    plt.xlabel("feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.title(model.__class__.__name__)
    
plot_feature_importance_datasets(model)
plt.show()



# subplot

import matplotlib.pyplot as plt

def plot_feature_importance_datasets(model, ax, model_name):
    n_features = datasets.data.shape[1]
    ax.barh(np.arange(n_features), model.feature_importances_, align='center')
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels([f'Feature {i}' for i in range(n_features)])
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_title(model_name)
    ax.set_ylim(-1, n_features)

# 서브플롯 구성 (2행 2열)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for i, model in enumerate(models):
    row, col = divmod(i, 2)
    plot_feature_importance_datasets(model, axes[row][col], model.__class__.__name__)

plt.tight_layout()
plt.show()

    