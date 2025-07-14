# 1. 라이브러리 로딩
import pandas as pd
import numpy as np
import random
import os
import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# 2. Seed 고정
SEED = 190
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# 3. Timestamped folder 생성
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
base_path = '/Users/jaewoo000/Desktop/IBM:Redhat/Study25/_data/dacon/cancer/'
save_dir = os.path.join(base_path, f"results_{timestamp}")
os.makedirs(save_dir, exist_ok=True)

# 4. 데이터 로딩
train_csv = pd.read_csv(base_path+'train.csv', index_col=0)
test_csv = pd.read_csv(base_path+'test.csv', index_col=0)
submission = pd.read_csv(base_path+'sample_submission.csv', index_col=0)

##################### 결측치 확인 ####################
#print(train_csv.info())
#print(train_csv.isnull().sum()) #결측치 없음
#print(test_csv.isna().sum()) #결측치 없음
# print(train_csv.describe())
# print(train_csv.shape, test_csv.shape) 

# 5. 전처리
train_csv['is_train'] = 1
test_csv['is_train'] = 0
combined = pd.concat([train_csv, test_csv], axis=0)

# 원핫인코딩
combined = pd.get_dummies(combined, columns=['Gender','Country','Race','Family_Background','Radiation_History',
                                             'Iodine_Deficiency','Smoke','Weight_Risk','Diabetes'],
                          drop_first=True, dtype=int)

# 불필요 칼럼 제거
drop_features = ["T3_Result","T4_Result","TSH_Result","Nodule_Size","Age"]
combined.drop(columns=drop_features, inplace=True)

# 다시 분리
train_csv = combined[combined['is_train']==1].drop(columns='is_train')
test_csv = combined[combined['is_train']==0].drop(columns=['is_train','Cancer'])

x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']

# 6. Train/Test Split + SMOTE
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, stratify=y, random_state=334)

smote = SMOTE(random_state=SEED)
x_train, y_train = smote.fit_resample(x_train, y_train)

# 7. KFold Ensemble with ModelCheckpoint
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
test_preds = np.zeros(test_csv.shape[0])
val_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(x, y)):
    print(f"\n🚀 Fold {fold+1}")

    x_tr, x_va = x.iloc[train_idx], x.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    # SMOTE 각 fold에 적용
    x_tr, y_tr = smote.fit_resample(x_tr, y_tr)

    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=SEED,
        eval_metric='logloss',
        use_label_encoder=False
    )

    model.fit(
        x_tr, y_tr,
        eval_set=[(x_va, y_va)],
        early_stopping_rounds=50,
        verbose=50
    )

    # ✅ Save model weights for this fold in timestamped folder
    model_save_path = os.path.join(save_dir, f"xgb_fold{fold+1}.json")
    model.save_model(model_save_path)
    print(f"💾 Model weights saved to {model_save_path}")

    # Evaluate threshold for best F1
    val_pred = model.predict_proba(x_va)[:,1]
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.3, 0.7, 0.01):
        f1 = f1_score(y_va, (val_pred > th).astype(int))
        if f1 > best_f1:
            best_f1, best_th = f1, th

    print(f"✅ Fold {fold+1} Best F1: {best_f1:.4f} at threshold {best_th:.2f}")
    val_scores.append(best_f1)

    # Test prediction
    test_preds += (model.predict_proba(test_csv)[:,1] > best_th).astype(int) / kf.n_splits

# 8. Submission saved in timestamped folder
submission['Cancer'] = (test_preds > 0.5).astype(int)
submission_filename = os.path.join(save_dir, f"submission_final.csv")
submission.to_csv(submission_filename)

print(f"\n🎯 KFold Average F1: {np.mean(val_scores):.4f}")
print(f"✅ Submission saved as {submission_filename}")

# # 1. 라이브러리 로딩
# import pandas as pd
# import numpy as np
# import random
# import os
# import datetime
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import f1_score
# from imblearn.over_sampling import SMOTE
# from xgboost import XGBClassifier
# from scipy.optimize import minimize_scalar

# # 2. Seed 고정
# SEED = 190
# random.seed(SEED)
# np.random.seed(SEED)
# os.environ['PYTHONHASHSEED'] = str(SEED)

# # 3. Timestamped folder 생성
# timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
# base_path = '/Users/jaewoo000/Desktop/IBM:Redhat/Study25/_data/dacon/cancer/'
# save_dir = os.path.join(base_path, f"results_{timestamp}")
# os.makedirs(save_dir, exist_ok=True)

# # 4. 데이터 로딩
# train_csv = pd.read_csv(base_path+'train.csv', index_col=0)
# test_csv = pd.read_csv(base_path+'test.csv', index_col=0)
# submission = pd.read_csv(base_path+'sample_submission.csv', index_col=0)

# # 5. 전처리
# train_csv['is_train'] = 1
# test_csv['is_train'] = 0
# combined = pd.concat([train_csv, test_csv], axis=0)

# # 원핫인코딩
# combined = pd.get_dummies(combined, columns=['Gender','Country','Race','Family_Background','Radiation_History',
#                                              'Iodine_Deficiency','Smoke','Weight_Risk','Diabetes'],
#                           drop_first=True, dtype=int)

# # 불필요 칼럼 제거
# drop_features = ["T3_Result","T4_Result","TSH_Result","Nodule_Size","Age"]
# combined.drop(columns=drop_features, inplace=True)

# # 다시 분리
# train_csv = combined[combined['is_train']==1].drop(columns='is_train')
# test_csv = combined[combined['is_train']==0].drop(columns=['is_train','Cancer'])

# x = train_csv.drop(['Cancer'], axis=1)
# y = train_csv['Cancer']

# # 6. KFold Ensemble with SMOTE within folds + OOF
# kf = StratifiedKFold(n_splits=12, shuffle=True, random_state=SEED)
# test_preds = np.zeros(test_csv.shape[0])
# val_scores = []
# oof_preds = np.zeros(x.shape[0])
# oof_targets = y.values

# smote = SMOTE(random_state=SEED)

# for fold, (train_idx, val_idx) in enumerate(kf.split(x, y)):
#     print(f"\n🚀 Fold {fold+1}")

#     x_tr, x_va = x.iloc[train_idx], x.iloc[val_idx]
#     y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

#     # SMOTE 각 fold에 적용
#     x_tr, y_tr = smote.fit_resample(x_tr, y_tr)

#     model = XGBClassifier(
#         n_estimators=1000,
#         learning_rate=0.05,
#         max_depth=5,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         reg_alpha=0.1,
#         reg_lambda=1,
#         random_state=SEED,
#         eval_metric='logloss',
#         use_label_encoder=False
#     )

#     model.fit(
#         x_tr, y_tr,
#         eval_set=[(x_va, y_va)],
#         early_stopping_rounds=50,
#         verbose=50
#     )

#     # ✅ Save model weights for this fold in timestamped folder
#     model_save_path = os.path.join(save_dir, f"xgb_fold{fold+1}.json")
#     model.save_model(model_save_path)
#     print(f"💾 Model weights saved to {model_save_path}")

#     # OOF prediction 저장
#     val_pred = model.predict_proba(x_va)[:,1]
#     oof_preds[val_idx] = val_pred

#     # Fold validation F1 체크 (threshold=0.5)
#     f1 = f1_score(y_va, (val_pred > 0.5).astype(int))
#     print(f"✅ Fold {fold+1} F1 (th=0.5): {f1:.4f}")
#     val_scores.append(f1)

#     # Test prediction (average probabilities)
#     test_preds += model.predict_proba(test_csv)[:,1] / kf.n_splits

# # 7. OOF 기반 최적 threshold 찾기 (scipy)
# def f1_opt(threshold):
#     return -f1_score(oof_targets, (oof_preds > threshold).astype(int))

# res = minimize_scalar(f1_opt, bounds=(0.3,0.7), method='bounded')
# best_th = res.x
# print(f"\n🔥 Optimized threshold via scipy: {best_th:.4f}, F1: {-res.fun:.4f}")

# # 8. 최적 threshold를 test prediction에 적용
# submission['Cancer'] = (test_preds > best_th).astype(int)

# # 9. Submission saved in timestamped folder
# submission_filename = os.path.join(save_dir, f"submission_final.csv")
# submission.to_csv(submission_filename)

# print(f"\n🎯 KFold Average F1 (th=0.5): {np.mean(val_scores):.4f}")
# print(f"✅ Submission saved as {submission_filename}")