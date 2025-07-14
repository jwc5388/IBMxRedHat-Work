
from keras.datasets import mnist, cifar10, cifar100
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import warnings
warnings.filterwarnings('ignore')

import os

# 기본 경로 설정
if os.path.exists('/workspace/TensorJae/Study25/'):
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')

np_path = os.path.join(BASE_PATH, '_save/save_npy/')

# 저장된 npy 파일 불러오기
x_train = np.load(np_path + 'keras44cd_x_train.npy')  # (25000, 150, 150, 3)
y_train = np.load(np_path + 'keras44cd_y_train.npy')
x_test = np.load(np_path + 'keras44cd_x_test.npy')    # (12500, 150, 150, 3)
y_test = np.load(np_path + 'keras44cd_y_test.npy')

print(f"x_train shape: {x_train.shape}")
print(f"x_test  shape: {x_test.shape}")

# Train + Test 결합
x = np.concatenate([x_train, x_test], axis=0)  # (37500, 150, 150, 3)
y = np.concatenate([y_train, y_test], axis=0)


print(x.shape)  # (37500, 150, 150, 3)
# exit()

# exit()

# reshape
x = x.reshape(x.shape[0], 150*150*3)  # (37500, 67500)                

# 3. 스케일링 (LDA 전에 추천)
# scaler = StandardScaler()
# x = scaler.fit_transform(x)

# 4. train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42, stratify=y)

# 5. LDA 적용
n_classes = len(np.unique(y))  #  10개 클래스
print(len(np.unique(y)))
# exit()
lda = LinearDiscriminantAnalysis(n_components=n_classes - 1)
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)

print("원본 x_train shape:", x_train.shape)
print("LDA 적용 후 shape:", x_train_lda.shape)  

# 원본 x_train shape: (48000, 3072)
# LDA 적용 후 shape: (48000, 9)

# 6. 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(x_train_lda, y_train)

# 7. 평가
y_pred = model.predict(x_test_lda)
acc = accuracy_score(y_test, y_pred)
print(f"\nLDA accuracy: {acc:.4f}")


