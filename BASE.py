import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)


# # path =basepath + '_data/'
# path = basepath + '_save/'

import numpy as np
from sklearn.utils.multiclass import type_of_target

def check_target_type(y):
    target_type = type_of_target(y)
    unique_classes = np.unique(y)

    print(f"🎯 데이터 타입: {target_type}")

    if target_type == 'binary':
        print("✅ 이진 분류 (binary classification)")
        print(f"클래스 목록: {unique_classes}")
    
    elif target_type == 'multiclass':
        print("✅ 다중 분류 (multi-class classification)")
        print(f"클래스 개수: {len(unique_classes)}")
        print(f"클래스 목록: {unique_classes}")
    
    elif target_type == 'multilabel-indicator':
        print("✅ 다중 레이블 분류 (multi-label classification)")
        print(f"데이터 shape: {y.shape} → 각 샘플마다 {y.shape[1]}개의 클래스 가능성")
    
    elif target_type == 'continuous':
        print("📉 회귀 문제 (continuous target)")
    
    else:
        print("⚠️ 인식되지 않는 형태입니다. 확인이 필요합니다.")
        
        
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
x, y = data.data, data.target

check_target_type(y)
