import numpy as np
import os
import time
import datetime
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
if os.path.exists('/workspace/TensorJae/Study25/'):
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
basepath = os.path.join(BASE_PATH)
path = basepath + '_data/diabetes/'

# 시간 문자열 생성
date = datetime.datetime.now().strftime("%m%d_%H%M")

# 데이터 불러오기
x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.8)

print(x_train.shape, y_train.shape)

# 모델 생성 함수
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8):
    inputs = Input(shape=(10,))
    x = Dense(node1, activation=activation)(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation)(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation)(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation)(x)
    x = Dense(node5, activation=activation)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# 하이퍼파라미터 생성
def create_hyperparameter():
    return {
        'batch_size': [32, 16, 8, 1, 64],
        'optimizer': ['adam', 'rmsprop', 'adadelta'],
        'drop': [0.2, 0.3, 0.4, 0.5],
        'activation': ['relu', 'elu', 'selu', 'linear'],
        'node1': [128, 64, 32, 16],
        'node2': [128, 64, 32, 16],
        'node3': [128, 64, 32, 16],
        'node4': [128, 64, 32, 16],
        'node5': [128, 64, 32, 16, 8]
    }

hyperparameters = create_hyperparameter()

# 콜백 세팅
filename = f"{date}_" + '{epoch:04d}-{val_loss:.4f}.h5'
filepath = os.path.join(path, 'k69' + filename)

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=7, factor=0.8, verbose=1)
mcp = ModelCheckpoint(filepath=filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

# 모델 래핑
keras_model = KerasRegressor(build_fn=build_model, verbose=0)

# RandomizedSearchCV
model = RandomizedSearchCV(
    estimator=keras_model,
    param_distributions=hyperparameters,
    cv=3,
    n_iter=10,
    verbose=1
)

# 훈련
start = time.time()
model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    callbacks=[es, rlr, mcp]
)
end = time.time()

# 결과 출력
print("최적의 하이퍼파라미터:", model.best_params_)
print("최고의 훈련 score (CV):", model.best_score_)
print("테스트 score:", model.score(x_test, y_test))

y_pred = model.predict(x_test)
print("r2_score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("소요 시간:", round(end - start, 2), "초")