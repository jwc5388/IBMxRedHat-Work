# iris

# california

# cancer

# wine

from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import matplotlib.pyplot as plt

# 1. Data
seed = 123
random.seed(seed)
np.random.seed(seed)

dataset1 = load_wine()
x = dataset1.data
y = dataset1.target
print(x.shape, y.shape)  

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=seed, stratify=y)

# 2. Model
model1 = DecisionTreeClassifier(random_state=seed)
model2 = RandomForestClassifier(random_state=seed)
model3 = GradientBoostingClassifier(random_state=seed)
model4 = XGBClassifier(random_state=seed)

models = [model1, model2, model3, model4]

for model in models:
    model.fit(x_train, y_train)
    print("===============", model.__class__.__name__, "======================")
    print('acc:', model.score(x_test, y_test))
    print('feature_importances_:', model.feature_importances_)

# 3. Subplot 시각화
def plot_feature_importance_datasets(model, ax, model_name):
    n_features = dataset1.data.shape[1]
    ax.barh(np.arange(n_features), model.feature_importances_, align='center')
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(dataset1.feature_names)
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_title(model_name)
    ax.set_ylim(-1, n_features)

# 2행 2열 서브플롯 구성
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for i, model in enumerate(models):
    row, col = divmod(i, 2)
    plot_feature_importance_datasets(model, axes[row][col], model.__class__.__name__)

plt.tight_layout()
plt.show()