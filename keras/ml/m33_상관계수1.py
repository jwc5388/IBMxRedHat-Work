import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
print(dataset.feature_names)


x = dataset['data']
y = dataset.target

df = pd.DataFrame(x, columns= dataset.feature_names)

print(df)
df['Target'] = y
print(df)


print('========================상관관계 히트맵 짜잔=========================')

print(df.corr())


import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(data = df.corr(), square= True, annot= True, cbar=True)

plt.show()