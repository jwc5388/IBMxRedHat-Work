# # ✅ 수정 사항 반영:
# # - early_stopping_rounds 및 callbacks 전부 제거 → 모델 정의부로 이동
# # - XGBoost, LightGBM, CatBoost 모두 반영 완료
# #1 best working version that feature selection takes really long but so far best score.

# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
# from sklearn.linear_model import Ridge
# from sklearn.feature_selection import SelectFromModel
# import xgboost as xgb
# import lightgbm as lgb
# import catboost as cat
# from rdkit import Chem
# from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, AllChem
# from datetime import datetime
# import random
# import warnings
# import copy

# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# warnings.filterwarnings('ignore')

# BASE_PATH = '/workspace/TensorJae/Study25/' if os.path.exists('/workspace/TensorJae/Study25/') else os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
# path = os.path.join(BASE_PATH, '_data/dacon/drugs/')
# train = pd.read_csv(path + 'train.csv')
# test = pd.read_csv(path + 'test.csv')
# submission = pd.read_csv(path + 'sample_submission.csv')

# def get_molecule_descriptors(smiles):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None: return [0] * 2233
#         basic = [
#             Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHAcceptors(mol),
#             Descriptors.NumHDonors(mol), Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
#             Descriptors.NumAromaticRings(mol), Descriptors.NumHeteroatoms(mol), Descriptors.FractionCSP3(mol),
#             Descriptors.NumAliphaticRings(mol), Lipinski.NumAromaticHeterocycles(mol),
#             Lipinski.NumSaturatedHeterocycles(mol), Lipinski.NumAliphaticHeterocycles(mol),
#             Descriptors.HeavyAtomCount(mol), Descriptors.RingCount(mol), Descriptors.NOCount(mol),
#             Descriptors.NHOHCount(mol), Descriptors.NumRadicalElectrons(mol)
#         ]
#         morgan = [int(b) for b in AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()]
#         maccs = [int(b) for b in MACCSkeys.GenMACCSKeys(mol).ToBitString()]
#         return basic + morgan + maccs
#     except:
#         return [0] * 2233

# train['features'] = train['Canonical_Smiles'].apply(get_molecule_descriptors)
# test['features'] = test['Canonical_Smiles'].apply(get_molecule_descriptors)

# x_raw = np.array(train['features'].tolist())
# y = train['Inhibition'].values
# x_test_raw = np.array(test['features'].tolist())

# scaler = RobustScaler()
# x_scaled = scaler.fit_transform(x_raw)
# x_test_scaled = scaler.transform(x_test_raw)

# x_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=0.2, random_state=seed)

# def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
# def normalized_rmse(y_true, y_pred): return rmse(y_true, y_pred) / (np.max(y_true) - np.min(y_true))
# def pearson_correlation(y_true, y_pred): return np.clip(np.corrcoef(y_true, y_pred)[0, 1], 0, 1)
# def competition_score(y_true, y_pred): return 0.5 * (1 - min(normalized_rmse(y_true, y_pred), 1)) + 0.5 * pearson_correlation(y_true, y_pred)

# def create_xgb_model():
#     return xgb.XGBRegressor(
#         n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
#         colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, random_state=seed,
#         # tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0,
#         early_stopping_rounds=50, eval_metric="rmse"
#     )

# def create_lgb_model():
#     return lgb.LGBMRegressor(
#         n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6,
#         subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1,
#         random_state=seed, early_stopping=50
#     )

# def create_cat_model():
#     return cat.CatBoostRegressor(
#         iterations=500, learning_rate=0.05, depth=6,
#         l2_leaf_reg=3, random_seed=seed, verbose=0,
#         early_stopping_rounds=50
#     )

# xgb_model = create_xgb_model()
# xgb_model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=0)

# booster = xgb_model.get_booster()
# score_dict = booster.get_score(importance_type='gain')
# total_gain = sum(score_dict.values())
# score_list = [score_dict.get(f"f{i}", 0) / total_gain for i in range(x_train.shape[1])]
# thresholds = np.sort(score_list)

# max_score = -np.inf
# best_selection = None
# for threshold in thresholds:
#     selection = SelectFromModel(xgb_model, threshold=threshold, prefit=True)
#     selected_train = selection.transform(x_train)
#     selected_val = selection.transform(x_val)
#     if selected_train.shape[1] == 0: continue
#     temp_model = create_xgb_model()
#     temp_model.fit(selected_train, y_train, eval_set=[(selected_val, y_val)], verbose=0)
#     score = competition_score(y_val, temp_model.predict(selected_val))
#     if score > max_score:
#         max_score = score
#         best_selection = selection

# print(f"\n🔥 Best score after feature selection: {max_score:.4f}")

# x_selected = best_selection.transform(x_scaled)
# x_test_selected = best_selection.transform(x_test_scaled)
# x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(x_selected, y, test_size=0.2, random_state=seed)

# base_models = {
#     "XGBoost": create_xgb_model(),
#     "LightGBM": create_lgb_model(),
#     "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
#         min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=seed),
#     "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=5,
#         min_samples_leaf=2, random_state=seed),
#     "CatBoost": create_cat_model()
# }

# trained_models = {}
# best_score = -np.inf
# best_model_name = None

# for name, model in base_models.items():
#     print(f"\n{name} 모델 학습 중...")
#     m = copy.deepcopy(model)
#     if name in ["CatBoost", "XGBoost", "LightGBM"]:
#         m.fit(x_train_final, y_train_final, eval_set=[(x_val_final, y_val_final)], verbose=0)
#     else:
#         m.fit(x_train_final, y_train_final)

#     y_pred = m.predict(x_val_final)
#     score = competition_score(y_val_final, y_pred)
#     print(f"→ Score: {score:.4f}")
#     trained_models[name] = m
#     if score > best_score:
#         best_score = score
#         best_model_name = name

# stacking_model = StackingRegressor(
#     estimators=[(k.lower(), v) for k, v in trained_models.items()],
#     final_estimator=Ridge(), n_jobs=-1
# )

# stacking_model.fit(x_train_final, y_train_final)
# y_pred_stack = stacking_model.predict(x_val_final)
# stack_score = competition_score(y_val_final, y_pred_stack)
# print(f"→ Stacking | Score: {stack_score:.4f}")

# final_model = stacking_model if stack_score > best_score else trained_models[best_model_name]
# final_model.fit(x_selected, y)
# y_pred_test = final_model.predict(x_test_selected)

# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# filename = f"submission_final_{timestamp}.csv"
# submission['Inhibition'] = y_pred_test
# submission.to_csv(os.path.join(path, filename), index=False)
# print(f"\n✅ 예측 결과 저장 완료 → {filename}")

# print("\n📊 모델별 성능 비교")
# for name, model in trained_models.items():
#     y_pred_compare = model.predict(x_val_final)
#     print(f"{name:20} | RMSE: {rmse(y_val_final, y_pred_compare):.4f} | NRMSE: {normalized_rmse(y_val_final, y_pred_compare):.4f} | Pearson: {pearson_correlation(y_val_final, y_pred_compare):.4f} | Score: {competition_score(y_val_final, y_pred_compare):.4f}")

# print(f"{'StackingRegressor':20} | RMSE: {rmse(y_val_final, y_pred_stack):.4f} | NRMSE: {normalized_rmse(y_val_final, y_pred_stack):.4f} | Pearson: {pearson_correlation(y_val_final, y_pred_stack):.4f} | Score: {stack_score:.4f}")


	# 1.	RDKit 기반 피처 생성 (basic + Morgan + MACCS)
	# 2.	정규화 (StandardScaler) 적용 후 feature importance 계산
	# 3.	XGBoost gain 기준 feature importance → SelectFromModel으로 최적 threshold 탐색
	# 4.	최적 threshold 기준으로 불필요한 feature 제거
	# 5.	그 결과를 반영하여 XGBoost, LightGBM, CatBoost 등 개별 모델 학습
	# 6.	스태킹 모델 학습 및 최종 성능 비교
	# 7.	테스트셋 예측값 제출 저장까지 포함
 
 
 
 
 
 
 
 
 
 #2 Modified version with faster feature deletion, earlystopping gone
 
 # ✅ 수정 사항 반영:
# - early_stopping_rounds 및 callbacks 전부 제거 → 모델 정의부로 이동
# - XGBoost, LightGBM, CatBoost 모두 반영 완료
# - feature selection loop → 상위 30개 feature 기준으로만 탐색
# - train/val split 인덱스 고정으로 검증 누수 방지

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, AllChem
from datetime import datetime
import random
import warnings
import copy

#41 best 42 best online
seed = 5388
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



# 2. feature vector 생성
train['features'] = train['Canonical_Smiles'].apply(get_molecule_descriptors)
test['features'] = test['Canonical_Smiles'].apply(get_molecule_descriptors)

x_raw = np.array(train['features'].tolist())
x_test_raw = np.array(test['features'].tolist())
y = train['Inhibition'].values

# # 3. IterativeImputer(BayesianRidge)로 결측치 채움
# imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=seed)
# x_raw = imputer.fit_transform(x_raw)
# x_test_raw = imputer.transform(x_test_raw)


# ✅ 3. SimpleImputer로 결측치 평균 대체
imputer = SimpleImputer(strategy='mean')
x_raw = imputer.fit_transform(x_raw)
x_test_raw = imputer.transform(x_test_raw)

# 4. Scaling
scaler = RobustScaler()
x_scaled = scaler.fit_transform(x_raw)
x_test_scaled = scaler.transform(x_test_raw)

# ✅ 고정 인덱스를 통한 split
train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
x_train, x_val = x_scaled[train_idx], x_scaled[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
def normalized_rmse(y_true, y_pred): return rmse(y_true, y_pred) / (np.max(y_true) - np.min(y_true))
def pearson_correlation(y_true, y_pred): return np.clip(np.corrcoef(y_true, y_pred)[0, 1], 0, 1)
def competition_score(y_true, y_pred): return 0.5 * (1 - min(normalized_rmse(y_true, y_pred), 1)) + 0.5 * pearson_correlation(y_true, y_pred)

def create_xgb_model():
    return xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=seed,
        tree_method='hist',
        predictor='auto',
        objective='reg:squarederror',
        early_stopping_rounds=50,     # ✅ XGBoost만 모델 생성자에 설정
        eval_metric='rmse'
    )

def create_lgb_model():
    return lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=seed,
        objective='regression'
    )

def create_cat_model():
    return cat.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=seed,
        loss_function='RMSE',
        verbose=0
    )

xgb_model = create_xgb_model()
xgb_model.fit(
    x_train, y_train,
    eval_set=[(x_val, y_val)],  # ✅ 필수
    verbose=0
)
# ⛔ 원래는 이렇게 validation 포함 → Data Leakage 발생!
# xgb_model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=0)

booster = xgb_model.get_booster()
score_dict = booster.get_score(importance_type='gain')
total_gain = sum(score_dict.values())
score_list = [score_dict.get(f"f{i}", 0) / total_gain for i in range(x_train.shape[1])]


# 🔧 Top-K Feature 개수 실험
top_k_list = [30, 50, 70, 100, 150, 300, 500, 600, 610,1000,1100,1200,1300,1400,1500, 2000, 2200]  # 원하는 K 개수 리스트
score_by_k = {}

# feature importance 순으로 index 정렬
importance_sorted_idx = np.argsort(score_list)[::-1]

for k in top_k_list:
    top_k_idx = importance_sorted_idx[:k]
    
    x_train_k = x_train[:, top_k_idx]
    x_val_k = x_val[:, top_k_idx]

    model_k = create_xgb_model()
    model_k.fit(x_train_k, y_train, eval_set=[(x_val_k, y_val)], verbose=0)

    y_pred_k = model_k.predict(x_val_k)
    score_k = competition_score(y_val, y_pred_k)

    score_by_k[k] = score_k
    print(f"Top {k:<3} features → Score: {score_k:.4f}")

# 🏆 가장 좋은 K 선택
best_k = max(score_by_k, key=score_by_k.get)
print(f"\n🔥 Best Top-K = {best_k} → Score: {score_by_k[best_k]:.4f}")

# 최종 feature selector 저장 (나중에 stacking 등에서 사용 가능)
top_k_idx = importance_sorted_idx[:best_k]
x_selected = x_scaled[:, top_k_idx]
x_test_selected = x_test_scaled[:, top_k_idx] 

x_train_final = x_selected[train_idx]
x_val_final = x_selected[val_idx]
y_train_final = y[train_idx]
y_val_final = y[val_idx]

base_models = {
    "XGBoost": create_xgb_model(),
    "LightGBM": create_lgb_model(),
    # "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
    #     min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=seed),
    # "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=5,
    #     min_samples_leaf=2, random_state=seed),
    "CatBoost": create_cat_model()
}

# ✅ 베이스 모델 훈련 (기존 코드 유지)
trained_models = {}
val_preds = {}  # 각 모델의 validation 예측값 저장
best_score = -np.inf
best_model_name = None
for name, model in base_models.items():
    print(f"\n{name} 모델 학습 중...")
    m = copy.deepcopy(model)

    if name == "LightGBM":
        m.fit(x_train_final, y_train_final,
              eval_set=[(x_val_final, y_val_final)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    elif name == "CatBoost":
        m.fit(x_train_final, y_train_final,
              eval_set=(x_val_final, y_val_final),
              early_stopping_rounds=50,
              verbose=0)
    elif name == "XGBoost":
        m.fit(x_train_final, y_train_final,
              eval_set=[(x_val_final, y_val_final)],
              verbose=0)
    else:
        m.fit(x_train_final, y_train_final)

    y_pred = m.predict(x_val_final)
    val_preds[name] = y_pred
    score = competition_score(y_val_final, y_pred)
    print(f"→ Score: {score:.4f}")
    trained_models[name] = m
    if score > best_score:
        best_score = score
        best_model_name = name

# ✅ 직접 스태킹을 위한 메타 데이터 구성
val_meta_features = np.column_stack([val_preds[m] for m in trained_models])
meta_model = Ridge()
meta_model.fit(val_meta_features, y_val_final)

# ✅ 테스트 데이터 예측
test_meta_features = np.column_stack([
    model.predict(x_test_selected) for model in trained_models.values()
])
y_pred_test = meta_model.predict(test_meta_features)

# ✅ 예측 저장
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"submission_manualstack_{timestamp}.csv"
submission['Inhibition'] = y_pred_test
submission.to_csv(os.path.join(path, filename), index=False)
print(f"\n✅ 예측 결과 저장 완료 → {filename}")

# 📊 모델별 성능 비교
print("\n📊 모델별 성능 비교")
for name, model in trained_models.items():
    y_pred_compare = val_preds[name]
    print(f"{name:20} | RMSE: {rmse(y_val_final, y_pred_compare):.4f} | NRMSE: {normalized_rmse(y_val_final, y_pred_compare):.4f} | Pearson: {pearson_correlation(y_val_final, y_pred_compare):.4f} | Score: {competition_score(y_val_final, y_pred_compare):.4f}")

# 메타 모델 성능
y_pred_meta_val = meta_model.predict(val_meta_features)
stack_score = competition_score(y_val_final, y_pred_meta_val)
print(f"{'Manual Stacking':20} | RMSE: {rmse(y_val_final, y_pred_meta_val):.4f} | NRMSE: {normalized_rmse(y_val_final, y_pred_meta_val):.4f} | Pearson: {pearson_correlation(y_val_final, y_pred_meta_val):.4f} | Score: {stack_score:.4f}")