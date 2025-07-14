from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array #땡겨온 이미지를 수치와

import matplotlib.pyplot as plt
import numpy as np

path = '/Users/jaewoo/Desktop/IBM x RedHat/Study25/_data/image/me/'

img = load_img(path + 'me.jpeg', target_size = (100,100))

print(img)
print(type(img))
#PIL = Python Image Library

plt.imshow(img)
plt.show()  

arr = img_to_array(img)
print(type(arr))
print(arr.shape) #(100,100,3)


###3차원 -> 4차원 
# arr = arr.reshape(1,100,100,3)

img = np.expand_dims(arr, axis=0)
print(img.shape)    #(1, 100, 100, 3)


#me 폴더에 데이터를 npy로 저장하겠다

np.save(path + 'keras47_me.npy', arr=img)
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from xgboost import XGBClassifier, plot_importance

# # 데이터 불러오기
# data_path = './_data/dacon/cancer/'
# train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)

# # X, y 분리
# x = train_csv.drop(columns='Cancer')
# y = train_csv['Cancer']

# # 범주형 Label Encoding만 처리 (필수)
# from sklearn.preprocessing import LabelEncoder
# for col in x.columns:
#     if x[col].dtype == 'object':
#         le = LabelEncoder()
#         x[col] = le.fit_transform(x[col].astype(str))

# # XGBoost 학습
# xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
# xgb.fit(x, y)

# # 중요도 시각화
# plt.figure(figsize=(12, 8))
# plot_importance(xgb, importance_type='gain', max_num_features=30)
# plt.title("XGBoost Feature Importance (No Preprocessing)")
# plt.tight_layout()
# plt.show()
