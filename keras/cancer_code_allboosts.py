# ========================================
# 🔰 라이브러리
# ========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

seed = 190
random.seed(seed)
np.random.seed(seed)

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

# ========================================
# 🔰 데이터 로딩 및 전처리
# ========================================
path = './Study25/_data/dacon/cancer/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv', index_col=0)

# train/test 통합 후 get_dummies
train_csv['is_train'] = 1
test_csv['is_train'] = 0
combined = pd.concat([train_csv, test_csv], axis=0)

# 원핫인코딩
combined = pd.get_dummies(combined, columns=['Gender','Country','Race','Family_Background','Radiation_History',
                                             'Iodine_Deficiency','Smoke','Weight_Risk','Diabetes'], drop_first=True, dtype=int)

# 불필요한 칼럼 제거
drop_features = ["T3_Result","T4_Result","TSH_Result","Nodule_Size","Age"]
combined.drop(columns=drop_features, inplace=True)

# 다시 분리
train_csv = combined[combined['is_train']==1].drop(columns='is_train')
test_csv = combined[combined['is_train']==0].drop(columns=['is_train','Cancer'])

# x, y split
x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']

# ========================================
# 🔰 train/test split + SMOTE
# ========================================
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=334)

smote = SMOTE(random_state=seed)
x_train, y_train = smote.fit_resample(x_train, y_train)

print(f"✅ x_train: {x_train.shape}, x_test: {x_test.shape}")

# ========================================
# 🔰 모델 정의 함수
# ========================================
def train_and_evaluate_model(model, model_name):
    print(f"🔧 Training {model_name}...")

    if model_name == 'XGBoost':
        model.fit(
            x_train, y_train,
            eval_set=[(x_test, y_test)],
            early_stopping_rounds=50,
            verbose=100
        )
    elif model_name == 'LightGBM':
        model.fit(
            x_train, y_train,
            eval_set=[(x_test, y_test)],
            early_stopping_rounds=50,
            verbose=100
        )
    elif model_name == 'CatBoost':
        model.fit(
            x_train, y_train,
            eval_set=(x_test, y_test),
            early_stopping_rounds=50,
            verbose=100
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Validation Predict + Threshold Tuning
    y_val_pred_prob = model.predict_proba(x_test)[:,1]
    best_threshold, best_f1 = 0.5, 0
    for threshold in np.arange(0.3, 0.7, 0.01):
        preds = (y_val_pred_prob > threshold).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"✅ {model_name} Best F1: {best_f1:.4f} at threshold {best_threshold:.2f}")
    return model, best_threshold

# ========================================
# 🔰 XGBoost
# ========================================
xgb = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=2,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=seed,
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model, xgb_threshold = train_and_evaluate_model(xgb, 'XGBoost')

# ========================================
# 🔰 LightGBM
# ========================================
lgbm = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    class_weight='balanced',
    random_state=seed
)
lgbm_model, lgbm_threshold = train_and_evaluate_model(lgbm, 'LightGBM')

# ========================================
# 🔰 CatBoost
# ========================================
cat = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=5,
    eval_metric='Logloss',
    auto_class_weights='Balanced',
    random_state=seed,
    verbose=0
)
cat_model, cat_threshold = train_and_evaluate_model(cat, 'CatBoost')

# ========================================
# 🔰 앙상블 예측
# ========================================
print("🔮 Generating ensemble submission...")

y_submit_xgb = xgb_model.predict_proba(test_csv)[:,1]
y_submit_lgbm = lgbm_model.predict_proba(test_csv)[:,1]
y_submit_cat = cat_model.predict_proba(test_csv)[:,1]

# Simple average ensemble
y_submit_ensemble = (y_submit_xgb + y_submit_lgbm + y_submit_cat) / 3
submission_csv['Cancer'] = (y_submit_ensemble > 0.5).astype(int)

filename = f"./_save/dacon_cancer/submission_ensemble_{timestamp}.csv"
submission_csv.to_csv(filename)
print(f"✅ Submission saved to: {filename}")
