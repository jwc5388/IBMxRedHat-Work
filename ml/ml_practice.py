# # # # # # # from keras.models import Sequential, load_model
# # # # # # # from keras.layers import Dense, Conv1D, Conv2D, Flatten, BatchNormalization
# # # # # # # from sklearn.model_selection import train_test_split,  KFold, StratifiedKFold
# # # # # # # from sklearn.datasets import load_diabetes
# # # # # # # from sklearn.metrics import r2_score, mean_squared_error
# # # # # # # from keras.callbacks import EarlyStopping, ModelCheckpoint

# # # # # # # from sklearn.preprocessing import StandardScaler, MinMaxScaler
# # # # # # # import numpy as np
# # # # # # # import time
# # # # # # # from sklearn.svm import LinearSVC
# # # # # # # from sklearn.linear_model import LogisticRegression
# # # # # # # from sklearn.tree import DecisionTreeClassifier
# # # # # # # from sklearn.ensemble import RandomForestClassifier
# # # # # # # import pandas as pd
# # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # from sklearn.datasets import load_wine
# # # # # # # from keras.utils import to_categorical
# # # # # # # from sklearn.datasets import fetch_covtype



# # # # # # # model_list = [LinearSVC(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]


# # # # # # # x, y = fetch_covtype(return_X_y=True)

# # # # # # # x_train, x_test, y_train, y_test= train_test_split(x,y, train_size=0.2, random_state=42)

# # # # # # # for model in model_list:
# # # # # # #     model.fit(x_train, y_train)
# # # # # # #     result = model.score(x_test, y_test)
# # # # # # #     print(f" {type(model).__name__} Validation Accuracy: {result : .4f}")
    
    
# # # # # # from sklearn.datasets import fetch_california_housing
# # # # # # from sklearn.model_selection import train_test_split
# # # # # # from sklearn.preprocessing import RobustScaler
# # # # # # from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
# # # # # # from sklearn.metrics import r2_score
# # # # # # import numpy as np
# # # # # # import warnings
# # # # # # warnings.filterwarnings('ignore')

# # # # # # from sklearn.utils import all_estimators
# # # # # # import sklearn as sk

# # # # # # x , y = fetch_california_housing(return_X_y=True)

# # # # # # kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# # # # # # scaler = RobustScaler()
# # # # # # x = scaler.fit_transform(x)

# # # # # # allAlgorithms = all_estimators(type_filter='regressor')



# # # # # # max_score = 0
# # # # # # max_name = None
# # # # # # model_scores = []
# # # # # # for (name, algorithm) in allAlgorithms:
# # # # # #     try:
# # # # # #         model = algorithm()
# # # # # #         scores =cross_val_score(model, x, y, cv = kfold, scoring='r2')
# # # # # #         mean_score = np.mean(scores)
# # # # # #         model_scores.append((name, mean_score))
        
        
# # # # # #         print()


# # # # # from sklearn.datasets import fetch_california_housing
# # # # # from sklearn.tree import DecisionTreeClassifier
# # # # # from sklearn.ensemble import RandomForestClassifier
# # # # # from sklearn.model_selection import train_test_split
# # # # # from sklearn.ensemble import GradientBoostingClassifier
# # # # # from xgboost import XGBClassifier, XGBRegressor
# # # # # import random
# # # # # import numpy as np
# # # # # import pandas as pd
# # # # # from sklearn.metrics import accuracy_score, r2_score
# # # # # import matplotlib.pyplot as plt
# # # # # import xgboost as xgb

# # # # # seed = 42
# # # # # random.seed(seed)
# # # # # np.random.seed(seed)

# # # # # dataset = fetch_california_housing()
# # # # # x = dataset.data
# # # # # y = dataset['target']

# # # # # feature_names = dataset.feature_names

# # # # # print(feature_names)

# # # # # # print(y)

# # # # # x_train , x_test, y_train, y_test= train_test_split(x,y, random_state=42, train_size=0.8)

# # # # # es = xgb.callback.EarlyStopping(
# # # # #     rounds = 20,
# # # # #     data_name = 'validation_0'
# # # # # )

# # # # # model = XGBRegressor(
# # # # #     n_estimators = 300,
# # # # #     random_state = seed,
# # # # #     gamma = 0,
# # # # #     min_child_weight = 0,
# # # # #     reg_alpha = 0,
# # # # #     reg_lambda = 1,
# # # # #     callbacks = [es]
# # # # # )


# # # # # model.fit(x_train, y_train, eval_set =[(x_test, y_test)], verbose  =1)

# # # # # print('r2:', model.score(x_test, y_test))
# # # # # score_dict = model.get_booster().get_score(importance_type = 'gain')

# # # # # total = sum(score_dict.values())
# # # # # print(total)

# # # # # score_list = [v/total for v in score_dict.values()]


# # # # # thresholds= np.sort(score_list)

# # # # # score_df = pd.DataFrame({
# # # # #     'feature' : feature_names,
# # # # #     'gain' : list(score_dict.values()),
    
# # # # # }).sort_values(by='gain', ascending=True)

# # # # # print(score_df)

# # # # # from sklearn.feature_selection import SelectFromModel

# # # # # delete_columns = []
# # # # # max_r2 = 0

# # # # # for i in thresholds:
# # # # #     selection = SelectFromModel(model, threshold=i, prefit=True)
# # # # #     select_x_train = selection.transform(x_train)
# # # # #     select_x_test = selection.transform(x_test)
    
# # # # #     mask = selection.get_support()
    
    
# # # # #     not_select_features = [feature_names[j]
# # # # #                            for j, selected in enumerate(mask)
# # # # #                            if not selected ]
    
    
# # # # #     if select_x_train.shape[1] == 0:
# # # # #         print(f"⚠️ Skipped: No features selected at threshold {i:.4f}")
# # # # #         continue

# # # # #     print(f"Threshold: {i:.4f}, n_features: {select_x_train.shape[1]}")
    
    
# # # # #     select_model = XGBRegressor(
# # # # #         n_estimators = 300,
# # # # #         random_state = seed,
# # # # #         gamma = 0,
# # # # #         min_child_weight = 0,
# # # # #         reg_alpha = 0,
# # # # #         reg_lambda = 1,
        
# # # # #     )
    
# # # # #     select_model.fit(select_x_test, y_train, eval_set = [(select_x_test, y_test)])
    
    
# # # # #     select_y_pred = model.predict(select_x_test)
# # # # #     score = r2_score(y_test, select_y_pred)
    
# # # # #     print('Trech=%.3f, n=%d, acc: %.4f%%' %(i, select_x_train.shape[1], score*100))

# # # # #     print('columns to delete:::', not_select_features)
# # # # #     print('===================================================')
    
# # # # #     if score>= max_r2:
# # # # #         delete_columns = not_select_features
# # # # #         max_r2 = score
# # # # #         nn = select_x_train.shape[1]
        
        
# # # # # print('for max score deleted columns:', delete_columns)
# # # # # print('max r2 score:', max_r2)
# # # # # print('남은 컬럼수:', nn, '개')    


# # # # from sklearn.datasets import load_iris
# # # # from sklearn.tree import DecisionTreeClassifier
# # # # from sklearn.ensemble import RandomForestClassifier
# # # # from sklearn.model_selection import train_test_split
# # # # from sklearn.ensemble import GradientBoostingClassifier
# # # # from xgboost import XGBClassifier
# # # # import random
# # # # import numpy as np
# # # # import xgboost as xgb
# # # # import pandas as pd
# # # # from sklearn.metrics import accuracy_score, r2_score


# # # # seed = 42ㅌ

# # # # random.seed(seed)
# # # # np.random.seed(seed)

# # # # dataset = load_iris()
# # # # x = dataset.data
# # # # y = dataset['target']

# # # # x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=42)

# # # # es = xgb.callback.EarlyStopping(
# # # #     rounds = 50,
# # # #     metric_name = 'mlogloss',
# # # #     data_name = 'validation_0',
# # # # )

# # # # model = XGBClassifier(
# # # #     n_estimators = 500,
# # # #     random_state = seed,
# # # #     gamma = 0,
# # # #     min_child_weight = 0,
# # # #     reg_alpha = 0,
# # # #     reg_lambda = 1,
# # # #     eval_metric = 'mlogloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
# # # #                             # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
# # # #     callbacks = [es]
    
# # # #     )

# # # # model.fit(x_train, y_train, eval_set = [(x_test, y_test)], verbose=0)

# # # # print('acc:', model.score(x_test, y_test))
# # # # print(model.feature_importances_)


# # # # thresholds = np.sort(model.feature_importances_)
# # # # print(thresholds)

# # # # from sklearn.feature_selection import SelectFromModel

# # # # for i in thresholds:
# # # #     selection = SelectFromModel(model, threshold=i, prefit=False)
# # # #     select_x_train = selection.transform(x_train)
# # # #     select_x_test = selection.transform(x_test)
    
# # # #     select_model = XGBClassifier(
# # # #         n_estimators = 500,
# # # #         random_state = seed,
# # # #         gamma = 0,
# # # #         min_child_weight = 0,
# # # #         reg_alpha = 0,
# # # #         reg_lambda = 1,
# # # #         eval_metric = 'mlogloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
# # # #                                 # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
# # # #         callbacks = [es]
        
# # # #         )
    
# # # #     select_model.fit(select_x_train, y_train, eval_set = [(select_x_test, y_test)], verbose = 0)
    
# # # #     select_y_pred = select_model.predict(select_x_test)
# # # #     score = accuracy_score(select_y_pred, y_test)
# # # #     print('Trech=%.3f, n=%d, ACC: %.4f%%' %(i, select_x_train.shape[1], score*100))


# # # from sklearn.datasets import load_iris
# # # from sklearn.tree import DecisionTreeClassifier
# # # from sklearn.ensemble import RandomForestClassifier
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.ensemble import GradientBoostingClassifier
# # # from xgboost import XGBClassifier
# # # import random
# # # import numpy as np
# # # import xgboost as xgb
# # # import pandas as pd
# # # from sklearn.metrics import accuracy_score

# # # seed = 42
# # # random.seed(seed)
# # # np.random.seed(seed)

# # # dataset = load_iris()
# # # x = dataset.data
# # # y = dataset['target']

# # # x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=42)


# # # es = xgb.callback.EarlyStopping(
# # #     rounds = 50,
# # #     metric_name = 'mlogloss',
# # #     data_name = 'validation_0',
# # # )

# # # model = XGBClassifier(
# # #     n_estimators = 500,
# # #     random_state = seed,
# # #     gamma = 0,
# # #     min_child_weight = 0,
# # #     reg_alpha = 0,
# # #     reg_lambda = 1,
# # #     eval_metric = 'mlogloss',
# # #     callbacks = [es]
# # # )

# # # model.fit(x_train, y_train, eval_set = [(x_test, y_test)], verbose = 0)

# # # print('acc:', model.score(x_test, y_test))
# # # print(model.feature_importances_)

# # # thresholds = np.sort(model.feature_importances_)
# # # print(thresholds)

# # # from sklearn.feature_selection import SelectFromModel

# # # for i in thresholds:
# # #     selection = SelectFromModel(model, threshold=i, prefit = False)
# # #     select_x_train = selection.transform(x_train)
# # #     select_x_test = selection.transform(x_test)
    
# # #     select_model = XGBClassifier(
# # #         n_estimators = 500,
# # #         random_state = seed,
# # #         gamma = 0,
# # #         min_child_weight = 0,
# # #         reg_alpha = 0,
# # #         reg_lambda = 1, 
# # #         eval_metric = 'mlogloss',
# # #         callbacks = [es]
        
        
# # #     )
    
# # #     select_model.fit(select_x_train, y_train, eval_set = [(select_x_test, y_test)], verbose = 0)
# # import random
# # import numpy as np
# # from sklearn.datasets import load_iris

# # seed = 42    
# # random.seed(seed)
# # np.random.seed(seed)

# # dataset = load_iris()

# import numpy as np
# import pandas as pd
# import random

# from sklearn.datasets import fetch_covtype
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score

# from keras.models import Sequential
# from keras.layers import Dense

# import tensorflow as tf

# seed = 123
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)


# dataset = fetch_covtype()
# x = dataset.data
# y = dataset['target']

# print(x.shape, y.shape)
# print(np.unique(y, return_counts= True))

# x_train, x_test,  y_train, y_test= train_test_split(x,y, random_state=seed, train_size=0.8, shuffle=True, stratify=y)

# from imblearn.over_sampling import SMOTE
# import sklearn as sk

# smote = SMOTE(random_state=seed,
#               k_neighbors=5,
#               sampling_strategy='auto')


# x_train, y_train = smote.fit_resample(x_train, y_train)

# print(np.unique(y_train, return_counts= True))

# model = Sequential()
# model.add(Dense(10, input_shape = (x.shape[1],0)))
# model.add(Dense(7, activation='softmax'))

# model.compile(loss = 'sparse_categorical_crossentropy',
#               optimizer = 'adam',
#               metrics = ['acc'])

# from keras.callbacks import EarlyStopping

# es = EarlyStopping(
#     monitor = 'val_loss',
#     mode= 'auto',
#     restore_best_weights=True,
#     patience = 10
# )

# model.fit(x_train, y_train, epochs = 30, validation_split=0.2, batch_size=32)


# result = model.evaluate(x_test, y_test)
# print('')

# dfjahkfdjh

# 뙤ㅏ어리ㅏㅁ어림넝리ㅏㅣㅏㅓ



# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import datetime

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# import xgboost as xgb
# import lightgbm as lgb

# from rdkit import Chem
# from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, AllChem

# import warnings
# warnings.filterwarnings('ignore')

# # ✅ 경로 및 시드
# BASE_PATH = '/workspace/TensorJae/Study25/' if os.path.exists('/workspace/TensorJae/Study25/') else os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
# path = os.path.join(BASE_PATH, '_data/dacon/drugs/')
# seed = 42
# np.random.seed(seed)

# # ✅ 데이터 로딩
# print("데이터 로딩 중...")
# train = pd.read_csv(path + 'train.csv')
# test = pd.read_csv(path + 'test.csv')
# submission = pd.read_csv(path + 'sample_submission.csv')
# print(f"훈련 데이터 : {train.shape}, 테스트 데이터 : {test.shape}")

# # ✅ SMILES → Molecule Feature 추출 함수
# def get_molecule_descriptors(smiles):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return [0] * 2232

#         basic_descriptors = [
#             Descriptors.MolWt(mol),
#             Descriptors.MolLogP(mol),
#             Descriptors.NumHAcceptors(mol),
#             Descriptors.NumHDonors(mol),
#             Descriptors.TPSA(mol),
#             Descriptors.NumRotatableBonds(mol),
#             Descriptors.NumAromaticRings(mol),
#             Descriptors.NumHeteroatoms(mol),
#             Descriptors.FractionCSP3(mol),
#             Descriptors.NumAliphaticRings(mol),
#             Lipinski.NumAromaticHeterocycles(mol),
#             Lipinski.NumSaturatedHeterocycles(mol),
#             Lipinski.NumAliphaticHeterocycles(mol),
#             Descriptors.HeavyAtomCount(mol),
#             Descriptors.RingCount(mol),
#             Descriptors.NOCount(mol),
#             Descriptors.NHOHCount(mol),
#             Descriptors.NumRadicalElectrons(mol),
#         ]

#         morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
#         morgan_features = [int(bit) for bit in morgan_fp.ToBitString()]

#         maccs_fp = MACCSkeys.GenMACCSKeys(mol)
#         maccs_features = [int(bit) for bit in maccs_fp.ToBitString()]

#         return basic_descriptors + morgan_features + maccs_features
#     except:
#         return [0] * 2232

# # ✅ 분자 특성 추출
# print("분자 특성 추출 중...")
# train['features'] = train['Canonical_Smiles'].apply(get_molecule_descriptors)
# test['features'] = test['Canonical_Smiles'].apply(get_molecule_descriptors)

# X_train = np.array(train['features'].tolist())
# X_test = np.array(test['features'].tolist())
# y_train = train['Inhibition'].values

# # ✅ 스케일링 & 데이터 분할
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# X_train_final, X_val, y_train_final, y_val = train_test_split(
#     X_train_scaled, y_train, test_size=0.2, random_state=seed
# )

# # ✅ 평가 함수
# def normalized_rmse(y_true, y_pred):
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     return rmse / (np.max(y_true) - np.min(y_true))

# def pearson_correlation(y_true, y_pred):
#     corr = np.corrcoef(y_true, y_pred)[0, 1]
#     return np.clip(corr, 0, 1)

# def competition_score(y_true, y_pred):
#     nrmse = min(normalized_rmse(y_true, y_pred), 1)
#     pearson = pearson_correlation(y_true, y_pred)
#     return 0.5 * (1 - nrmse) + 0.5 * pearson

# # ✅ 모델 학습 및 평가 함수
# def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
#     model.fit(X_train, y_train)
#     y_val_pred = model.predict(X_val)
#     score = competition_score(y_val, y_val_pred)
#     print(f"[{model.__class__.__name__}] NRMSE: {normalized_rmse(y_val, y_val_pred):.4f}, Pearson: {pearson_correlation(y_val, y_val_pred):.4f}, Score: {score:.4f}")
#     return model, score

# # ✅ 모델 구성
# models = {
#     "XGBoost": xgb.XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=6,
#                                  subsample=0.8, colsample_bytree=0.8, gamma=0,
#                                  reg_alpha=0.1, reg_lambda=1, random_state=seed),
#     "LightGBM": lgb.LGBMRegressor(n_estimators=700, learning_rate=0.05, num_leaves=31,
#                                   max_depth=6, subsample=0.8, colsample_bytree=0.8,
#                                   reg_alpha=0.1, reg_lambda=1, random_state=seed),
#     "GradientBoosting": GradientBoostingRegressor(n_estimators=700, learning_rate=0.05,
#                                                   max_depth=5, min_samples_split=5,
#                                                   min_samples_leaf=2, subsample=0.8,
#                                                   random_state=seed),
#     "RandomForest": RandomForestRegressor(n_estimators=700, max_depth=10,
#                                           min_samples_split=5, min_samples_leaf=2,
#                                           random_state=seed)
# }

# best_score = -np.inf
# best_model_name = None
# trained_models = {}

# # ✅ 모델 학습 및 최적 선택
# for name, model in models.items():
#     print(f"\n{name} 학습 중...")
#     trained_model, score = train_and_evaluate_model(model, X_train_final, y_train_final, X_val, y_val)
#     trained_models[name] = trained_model
#     if score > best_score:
#         best_score = score
#         best_model_name = name

# print(f"\n🎯 최고 모델: {best_model_name} / Score: {best_score:.4f}")

# # ✅ 전체 학습 + 예측
# final_model = models[best_model_name]
# final_model.fit(X_train_scaled, y_train)
# test_preds = final_model.predict(X_test_scaled)

# # ✅ 결과 저장
# submission['Inhibition'] = test_preds
# # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# output_file = os.path.join(path, f'submission_{best_model_name}_{best_score}.csv')
# submission.to_csv(output_file, index=False)
# print(f"예측 결과 저장 완료: {output_file}")

# # ✅ 예측 시각화
# plt.figure(figsize=(8, 6))
# y_val_pred = final_model.predict(X_val)
# plt.scatter(y_val, y_val_pred, alpha=0.5)
# plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
# plt.xlabel('실제값')
# plt.ylabel('예측값')
# plt.title(f'{best_model_name} 모델 검증 성능')
# plt.savefig(os.path.join(path, 'model_performance.png'))
# plt.close()
# print("📈 검증 시각화 저장 완료")

# # ✅ Feature Importance (가능한 경우)
# if hasattr(final_model, 'feature_importances_'):
#     importances = final_model.feature_importances_
#     indices = np.argsort(importances)[::-1][:20]
#     plt.figure(figsize=(10, 8))
#     sns.barplot(x=importances[indices], y=[f'feat_{i}' for i in indices])
#     plt.title(f'Top 20 Feature Importances ({best_model_name})')
#     plt.xlabel('Importance')
#     plt.ylabel('Feature')
#     plt.tight_layout()
#     plt.savefig(os.path.join(path, 'feature_importance.png'))
#     plt.close()
#     print("📊 Feature Importance 저장 완료")



# ✅ 최종 성능 개선 포함 전체 코드
# 적용 사항:
# - log1p 타겟변환
# - SMILES 통계 feature 추가
# - RFE + XGB 기반 feature 선택
# - Permutation Importance 적용

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, AllChem
from datetime import datetime
import random
import warnings
import copy

seed = 42
random.seed(seed)
np.random.seed(seed)
warnings.filterwarnings('ignore')

BASE_PATH = '/workspace/TensorJae/Study25/' if os.path.exists('/workspace/TensorJae/Study25/') else os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
path = os.path.join(BASE_PATH, '_data/dacon/drugs/')
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')

def get_molecule_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return [np.nan] * 2233
        basic = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol), Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.NumHeteroatoms(mol), Descriptors.FractionCSP3(mol),
            Descriptors.NumAliphaticRings(mol), Lipinski.NumAromaticHeterocycles(mol),
            Lipinski.NumSaturatedHeterocycles(mol), Lipinski.NumAliphaticHeterocycles(mol),
            Descriptors.HeavyAtomCount(mol), Descriptors.RingCount(mol), Descriptors.NOCount(mol),
            Descriptors.NHOHCount(mol), Descriptors.NumRadicalElectrons(mol)
        ]
        morgan = [int(b) for b in AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()]
        maccs = [int(b) for b in MACCSkeys.GenMACCSKeys(mol).ToBitString()]
        return basic + morgan + maccs
    except:
        return [np.nan] * 2233

def get_smiles_stats(smiles):
    return [
        len(smiles), smiles.count('='), smiles.count('#'), smiles.count('('),
        smiles.count('['), smiles.count('Cl'), smiles.count('Br')
    ]

train['features'] = train['Canonical_Smiles'].apply(get_molecule_descriptors)
test['features'] = test['Canonical_Smiles'].apply(get_molecule_descriptors)

train['smiles_stats'] = train['Canonical_Smiles'].apply(get_smiles_stats)
test['smiles_stats'] = test['Canonical_Smiles'].apply(get_smiles_stats)

x_raw = np.array(train['features'].tolist())
x_test_raw = np.array(test['features'].tolist())
y_raw = train['Inhibition'].values

x_raw = np.hstack([x_raw, np.array(train['smiles_stats'].tolist())])
x_test_raw = np.hstack([x_test_raw, np.array(test['smiles_stats'].tolist())])

# ✅ log1p 변환
y = np.log1p(y_raw)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
x_raw = imputer.fit_transform(x_raw)
x_test_raw = imputer.transform(x_test_raw)

scaler = RobustScaler()
x_scaled = scaler.fit_transform(x_raw)
x_test_scaled = scaler.transform(x_test_raw)

train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
x_train, x_val = x_scaled[train_idx], x_scaled[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
def normalized_rmse(y_true, y_pred): return rmse(y_true, y_pred) / (np.max(y_true) - np.min(y_true))
def pearson_correlation(y_true, y_pred): return np.clip(np.corrcoef(y_true, y_pred)[0, 1], 0, 1)
def competition_score(y_true, y_pred): return 0.5 * (1 - min(normalized_rmse(y_true, y_pred), 1)) + 0.5 * pearson_correlation(y_true, y_pred)

def create_xgb_model():
    return xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, random_state=seed, tree_method='hist')

# ✅ RFE 기반 feature 선택
selector = RFE(create_xgb_model(), n_features_to_select=600, step=100)
selector.fit(x_train, y_train)
x_train_sel = selector.transform(x_train)
x_val_sel = selector.transform(x_val)
x_scaled_sel = selector.transform(x_scaled)
x_test_sel = selector.transform(x_test_scaled)

# ✅ Permutation Importance 적용하여 중요도 재정렬
perm = permutation_importance(create_xgb_model().fit(x_train_sel, y_train), x_val_sel, y_val, n_repeats=5, random_state=seed)
perm_idx = np.argsort(perm.importances_mean)[::-1][:600]
x_train_sel = x_train_sel[:, perm_idx]
x_val_sel = x_val_sel[:, perm_idx]
x_scaled_sel = x_scaled_sel[:, perm_idx]
x_test_sel = x_test_sel[:, perm_idx]

x_train_final = x_train_sel
x_val_final = x_val_sel
y_train_final = y_train
y_val_final = y_val

base_models = {
    "XGBoost": create_xgb_model(),
    "LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=seed),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=seed),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=seed),
    "CatBoost": cat.CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_seed=seed, verbose=0)
}

trained_models = {}
best_score = -np.inf
best_model_name = None

for name, model in base_models.items():
    print(f"\n{name} 모델 학습 중...")
    m = copy.deepcopy(model)
    m.fit(x_train_final, y_train_final)
    y_pred = m.predict(x_val_final)
    score = competition_score(y_val_final, y_pred)
    print(f"→ Score: {score:.4f}")
    trained_models[name] = m
    if score > best_score:
        best_score = score
        best_model_name = name

stacking_model = StackingRegressor(
    estimators=[(k.lower(), v) for k, v in trained_models.items()],
    final_estimator=Ridge(), n_jobs=-1
)

stacking_model.fit(x_train_final, y_train_final)
y_pred_stack = stacking_model.predict(x_val_final)
stack_score = competition_score(y_val_final, y_pred_stack)
print(f"→ Stacking | Score: {stack_score:.4f}")

final_model = stacking_model if stack_score > best_score else trained_models[best_model_name]
final_model.fit(x_scaled_sel, y)
y_pred_test = final_model.predict(x_test_sel)
y_pred_test = np.expm1(y_pred_test)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"submission_final_{timestamp}.csv"
submission['Inhibition'] = y_pred_test
submission.to_csv(os.path.join(path, filename), index=False)
print(f"\n✅ 예측 결과 저장 완료 → {filename}")