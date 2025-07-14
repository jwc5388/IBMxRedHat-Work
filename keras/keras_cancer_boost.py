






# # # # # =========================
# # # # # 📦 Import Libraries
# # # # # =========================
# # # # import numpy as np
# # # # import pandas as pd
# # # # import os
# # # # from datetime import datetime

# # # # from sklearn.model_selection import StratifiedKFold
# # # # from sklearn.linear_model import LogisticRegression
# # # # from sklearn.metrics import f1_score
# # # # from sklearn.preprocessing import MinMaxScaler
# # # # from sklearn.ensemble import StackingClassifier

# # # # from imblearn.over_sampling import SMOTE

# # # # from keras.models import Sequential
# # # # from keras.layers import Dense, Dropout, BatchNormalization
# # # # from keras.wrappers.scikit_learn import KerasClassifier
# # # # from keras.callbacks import EarlyStopping

# # # # from xgboost import XGBClassifier
# # # # import lightgbm as lgb

# # # # # =========================
# # # # # 📂 Load Data
# # # # # =========================
# # # # data_path = '/Users/jaewoo/Desktop/IBM x RedHat/Study25/_data/dacon/cancer/'

# # # # train = pd.read_csv(data_path + 'train.csv', index_col=0)
# # # # test = pd.read_csv(data_path + 'test.csv', index_col=0)
# # # # submission = pd.read_csv(data_path + 'sample_submission.csv', index_col=0)

# # # # x = train.drop('Cancer', axis=1)
# # # # y = train['Cancer']
# # # # x_test = test.copy()

# # # # # =========================
# # # # # 🧼 Preprocessing
# # # # # =========================
# # # # # One-hot encode categorical features
# # # # x = pd.get_dummies(x)
# # # # x_test = pd.get_dummies(x_test)
# # # # x, x_test = x.align(x_test, join='left', axis=1, fill_value=0)

# # # # # Normalize numerical features
# # # # scaler = MinMaxScaler()
# # # # x_scaled = scaler.fit_transform(x)
# # # # x_test_scaled = scaler.transform(x_test)

# # # # # =========================
# # # # # 🧠 Keras Model Definition
# # # # # =========================
# # # # def build_keras_model():
# # # #     model = Sequential()
# # # #     model.add(Dense(128, input_dim=x_scaled.shape[1], activation='relu'))
# # # #     model.add(BatchNormalization())
# # # #     model.add(Dropout(0.3))
# # # #     model.add(Dense(64, activation='relu'))
# # # #     model.add(BatchNormalization())
# # # #     model.add(Dropout(0.2))
# # # #     model.add(Dense(1, activation='sigmoid'))
# # # #     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[])
# # # #     return model

# # # # # Wrap Keras model with Scikit-learn interface
# # # # keras_model = KerasClassifier(build_fn=build_keras_model, epochs=50, batch_size=32, verbose=0)

# # # # # =========================
# # # # # 🔗 Define Base Models
# # # # # =========================
# # # # base_models = [
# # # #     ('keras', keras_model),
# # # #     ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
# # # #     ('lgb', lgb.LGBMClassifier(random_state=42))
# # # # ]

# # # # # =========================
# # # # # 🔄 Meta Model
# # # # # =========================
# # # # meta_model = LogisticRegression(max_iter=1000)

# # # # # =========================
# # # # # 🔀 Build Stacking Classifier
# # # # # =========================
# # # # stack_model = StackingClassifier(
# # # #     estimators=base_models,
# # # #     final_estimator=meta_model,
# # # #     passthrough=True,
# # # #     cv=5,
# # # #     n_jobs=-1
# # # # )

# # # # # =========================
# # # # # ⚖️ Handle Class Imbalance with SMOTE
# # # # # =========================
# # # # x_resampled, y_resampled = SMOTE(random_state=42).fit_resample(x_scaled, y)

# # # # # =========================
# # # # # 🏋️ Train Stacking Ensemble
# # # # # =========================
# # # # stack_model.fit(x_resampled, y_resampled)

# # # # # =========================
# # # # # 🔍 Threshold Optimization (F1 기준)
# # # # # =========================
# # # # val_preds = stack_model.predict_proba(x_scaled)[:, 1]
# # # # thresholds = np.linspace(0.1, 0.9, 100)
# # # # f1s = [f1_score(y, (val_preds > t).astype(int)) for t in thresholds]
# # # # best_thresh = thresholds[np.argmax(f1s)]
# # # # print(f"✅ Best F1 Score: {max(f1s):.4f} at Threshold: {best_thresh:.2f}")

# # # # # =========================
# # # # # 🧾 Test Set Prediction
# # # # # =========================
# # # # y_submit = (stack_model.predict_proba(x_test_scaled)[:, 1] > best_thresh).astype(int)
# # # # submission['Cancer'] = y_submit

# # # # # =========================
# # # # # 💾 Save Submission
# # # # # =========================
# # # # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # save_path = f'/Users/jaewoo/Desktop/IBM x RedHat/Study25/_save/dacon_cancer/stacking_{timestamp}.csv'
# # # # submission.to_csv(save_path)
# # # # print(f"📁 Submission saved to: {save_path}")








# # # # # === Import Libraries ===
# # # # import pandas as pd
# # # # import numpy as np
# # # # import os
# # # # from datetime import datetime
# # # # from keras.models import Sequential
# # # # from keras.layers import Dense, Dropout, BatchNormalization
# # # # from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# # # # from keras.regularizers import l2
# # # # from keras.metrics import AUC, Precision, Recall

# # # # from sklearn.model_selection import StratifiedShuffleSplit
# # # # from sklearn.preprocessing import MinMaxScaler
# # # # from sklearn.metrics import f1_score
# # # # from xgboost import XGBClassifier

# # # # # === Set Random Seed for Reproducibility ===
# # # # np.random.seed(333)

# # # # # === Set Save Path ===
# # # # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # base_path = f'/Users/jaewoo/Desktop/IBM x RedHat/Study25/_save/dacon_cancer/{timestamp}'
# # # # os.makedirs(base_path, exist_ok=True)

# # # # # === Load Data ===
# # # # data_path = '/Users/jaewoo/Desktop/IBM x RedHat/Study25/_data/dacon/cancer/'
# # # # train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
# # # # test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
# # # # submission_csv = pd.read_csv(data_path + 'sample_submission.csv', index_col=0)

# # # # # === Preprocessing ===
# # # # x = train_csv.drop(['Cancer'], axis=1)
# # # # y = train_csv['Cancer']

# # # # # Categorical columns handling without One-Hot Encoding
# # # # x = pd.get_dummies(x, drop_first=True)
# # # # test_csv = pd.get_dummies(test_csv, drop_first=True)
# # # # x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)

# # # # # === Scaling ===
# # # # scaler = MinMaxScaler()
# # # # x_scaled = scaler.fit_transform(x)
# # # # test_scaled = scaler.transform(test_csv)

# # # # # === Feature Selection with XGBoost ===
# # # # xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
# # # # xgb.fit(x_scaled, y)
# # # # importances = xgb.feature_importances_
# # # # threshold = np.percentile(importances, 25)
# # # # selected_indices = np.where(importances > threshold)[0]
# # # # x_selected = x_scaled[:, selected_indices]
# # # # test_selected = test_scaled[:, selected_indices]

# # # # # === Train/Validation Split ===
# # # # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# # # # for train_idx, val_idx in sss.split(x_selected, y):
# # # #     x_train, x_val = x_selected[train_idx], x_selected[val_idx]
# # # #     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# # # # # === Model Builder Function ===
# # # # def build_model(input_dim):
# # # #     model = Sequential([
# # # #         Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.001)),
# # # #         BatchNormalization(), Dropout(0.2),
# # # #         Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
# # # #         BatchNormalization(), Dropout(0.3),
# # # #         Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
# # # #         BatchNormalization(), Dropout(0.2),
# # # #         Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
# # # #         BatchNormalization(), Dropout(0.2),
# # # #         Dense(1, activation='sigmoid')
# # # #     ])
# # # #     model.compile(loss='binary_crossentropy', optimizer='adam',
# # # #                   metrics=['accuracy', AUC(name='auc'), Precision(), Recall()])
# # # #     return model

# # # # # === Callbacks ===
# # # # es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
# # # # lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

# # # # # === ModelCheckpoint (MCP) ===
# # # # model_path = os.path.join(base_path, f"best_model_{timestamp}.h5")
# # # # mcp = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)

# # # # # === Train Ensemble ===
# # # # n_models = 3
# # # # models = []
# # # # val_preds = []

# # # # for i in range(n_models):
# # # #     print(f"\n🎯 Training model {i+1}/{n_models}")
# # # #     model = build_model(x_selected.shape[1])
# # # #     model.fit(x_train, y_train,
# # # #               validation_data=(x_val, y_val),
# # # #               epochs=77,
# # # #               batch_size=16,
# # # #               callbacks=[es, lr, mcp],  # Add ModelCheckpoint here
# # # #               verbose=1)
# # # #     models.append(model)
# # # #     val_preds.append(model.predict(x_val).ravel())

# # # # # === Average Validation Predictions ===
# # # # y_val_pred = np.mean(val_preds, axis=0)

# # # # # === Find Best Threshold ===
# # # # thresholds = np.arange(0.2, 0.8, 0.01)
# # # # f1_scores = [f1_score(y_val, (y_val_pred > t).astype(int)) for t in thresholds]
# # # # best_idx = np.argmax(f1_scores)
# # # # best_threshold = thresholds[best_idx]
# # # # best_f1 = f1_scores[best_idx]

# # # # # === Final Metrics from Best Model ===
# # # # loss, accuracy, auc, precision, recall = models[0].evaluate(x_val, y_val, verbose=0)
# # # # print(f'✅ loss: {loss:.4f}')
# # # # print(f'✅ acc : {accuracy:.4f}')
# # # # print(f'✅ AUC : {auc:.4f}')
# # # # print(f'✅ Precision: {precision:.4f}')
# # # # print(f'✅ Recall   : {recall:.4f}')
# # # # print(f'✅ Best F1: {best_f1:.4f} at threshold {best_threshold:.2f}')

# # # # # === Predict and Save Submission ===
# # # # test_preds = np.mean([model.predict(test_selected).ravel() for model in models], axis=0)
# # # # submission_csv['Cancer'] = (test_preds > best_threshold).astype(int)
# # # # submission_path = os.path.join(base_path, f'submission_{timestamp}.csv')
# # # # submission_csv.to_csv(submission_path)
# # # # print(f"✅ Submission saved to: {submission_path}")


# # # # === Import Libraries ===
# # # import pandas as pd
# # # import numpy as np
# # # import os
# # # from datetime import datetime
# # # from keras.models import Sequential, load_model
# # # from keras.layers import Dense, Dropout, BatchNormalization
# # # from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# # # from keras.regularizers import l2
# # # from keras.metrics import AUC, Precision, Recall

# # # from sklearn.model_selection import StratifiedShuffleSplit
# # # from sklearn.preprocessing import MinMaxScaler
# # # from sklearn.metrics import f1_score
# # # from sklearn.linear_model import LogisticRegression

# # # from xgboost import XGBClassifier
# # # import lightgbm as lgb
# # # from catboost import CatBoostClassifier

# # # # === Set Random Seed ===
# # # np.random.seed(333)

# # # # === Set Save Path ===
# # # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # base_path = f'./_save/dacon_cancer/{timestamp}'
# # # os.makedirs(base_path, exist_ok=True)

# # # # === Load Data ===
# # # data_path = './_data/dacon/cancer/'
# # # train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
# # # test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
# # # submission_csv = pd.read_csv(data_path + 'sample_submission.csv', index_col=0)

# # # # === Preprocessing ===
# # # x = train_csv.drop(['Cancer'], axis=1)
# # # y = train_csv['Cancer']
# # # x = pd.get_dummies(x, drop_first=True)
# # # test_csv = pd.get_dummies(test_csv, drop_first=True)
# # # x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)

# # # # === Scaling ===
# # # scaler = MinMaxScaler()
# # # x_scaled = scaler.fit_transform(x)
# # # test_scaled = scaler.transform(test_csv)

# # # # === Feature Selection ===
# # # xgb_fs = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
# # # xgb_fs.fit(x_scaled, y)
# # # importances = xgb_fs.feature_importances_
# # # threshold = np.percentile(importances, 25)
# # # selected_indices = np.where(importances > threshold)[0]
# # # x_selected = x_scaled[:, selected_indices]
# # # test_selected = test_scaled[:, selected_indices]

# # # # === Train/Validation Split ===
# # # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# # # for train_idx, val_idx in sss.split(x_selected, y):
# # #     x_train, x_val = x_selected[train_idx], x_selected[val_idx]
# # #     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# # # # === Neural Network Model Builder ===
# # # def build_model(input_dim):
# # #     model = Sequential([
# # #         Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.001)),
# # #         BatchNormalization(), Dropout(0.2),
# # #         Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
# # #         BatchNormalization(), Dropout(0.3),
# # #         Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
# # #         BatchNormalization(), Dropout(0.2),
# # #         Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
# # #         BatchNormalization(), Dropout(0.2),
# # #         Dense(1, activation='sigmoid')
# # #     ])
# # #     model.compile(loss='binary_crossentropy', optimizer='adam',
# # #                   metrics=['accuracy', AUC(name='auc'), Precision(), Recall()])
# # #     return model

# # # # === Callbacks (shared) ===
# # # es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
# # # lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

# # # # === Train Neural Network Ensemble ===
# # # n_models = 3
# # # val_preds_nn = []
# # # models = []

# # # for i in range(n_models):
# # #     print(f"\n🔁 Training NN model {i+1}/{n_models}")
# # #     model = build_model(x_selected.shape[1])
    
# # #     # ModelCheckpoint to save best weights
# # #     model_path = os.path.join(base_path, f"nn_model_{i}.h5")
# # #     mcp = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)

# # #     model.fit(x_train, y_train,
# # #               validation_data=(x_val, y_val),
# # #               epochs=100, batch_size=16,
# # #               callbacks=[es, lr, mcp],
# # #               verbose=1)
    
# # #     # Load best weights
# # #     best_model = load_model(model_path)
# # #     models.append(best_model)
# # #     val_preds_nn.append(best_model.predict(x_val).ravel())

# # # val_pred_nn_avg = np.mean(val_preds_nn, axis=0)

# # # # === Train Boosting Models ===
# # # print("⚡ Training XGBoost...")
# # # xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
# # # xgb_model.fit(x_train, y_train)
# # # xgb_val_pred = xgb_model.predict_proba(x_val)[:, 1]


# # # from lightgbm import early_stopping, log_evaluation
# # # print("⚡ Training LightGBM...")
# # # lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
# # # lgb_model.fit(
# # #     x_train, y_train,
# # #     eval_set=[(x_val, y_val)],
# # #     callbacks=[
# # #         early_stopping(10),
# # #         log_evaluation(0)
# # #     ]
# # # )
# # # lgb_val_pred = lgb_model.predict_proba(x_val)[:, 1]

# # # print("⚡ Training CatBoost...")
# # # cat_model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_state=42)
# # # cat_model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=20)
# # # cat_val_pred = cat_model.predict_proba(x_val)[:, 1]

# # # # === Meta-Model Stacking ===
# # # print("🔗 Stacking models...")
# # # stacked_val = np.column_stack([val_pred_nn_avg, xgb_val_pred, lgb_val_pred, cat_val_pred])
# # # meta_model = LogisticRegression()
# # # meta_model.fit(stacked_val, y_val)
# # # stacked_pred = meta_model.predict(stacked_val)

# # # # === F1 Score Evaluation ===
# # # best_f1 = f1_score(y_val, stacked_pred)
# # # print(f"🏆 Final Stacked F1: {best_f1:.4f}")

# # # # === Predict Test Data ===
# # # print("🚀 Predicting on test set...")
# # # nn_test_preds = [model.predict(test_selected).ravel() for model in models]
# # # xgb_test_pred = xgb_model.predict_proba(test_selected)[:, 1]
# # # lgb_test_pred = lgb_model.predict_proba(test_selected)[:, 1]
# # # cat_test_pred = cat_model.predict_proba(test_selected)[:, 1]
# # # stacked_test = np.column_stack([np.mean(nn_test_preds, axis=0), xgb_test_pred, lgb_test_pred, cat_test_pred])
# # # final_test_pred = meta_model.predict(stacked_test)

# # # # === Save Submission ===
# # # submission_csv['Cancer'] = final_test_pred
# # # submission_path = os.path.join(base_path, f'submission_boosted_{timestamp}.csv')
# # # submission_csv.to_csv(submission_path)
# # # print(f"✅ Submission saved: {submission_path}")











# # # === Import Libraries ===
# # import pandas as pd
# # import numpy as np
# # import os
# # from datetime import datetime
# # import optuna  # Optuna for hyperparameter tuning

# # # Keras and TensorFlow
# # import tensorflow as tf
# # from keras.models import Sequential, load_model
# # from keras.layers import Dense, Dropout, BatchNormalization
# # from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# # from keras.regularizers import l2
# # from keras.metrics import AUC, Precision, Recall

# # # Scikit-learn and Boosting
# # from sklearn.model_selection import StratifiedShuffleSplit
# # from sklearn.preprocessing import MinMaxScaler
# # from sklearn.metrics import f1_score, roc_auc_score
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.utils.class_weight import compute_class_weight
# # from xgboost import XGBClassifier
# # import lightgbm as lgb
# # from catboost import CatBoostClassifier
# # from lightgbm import early_stopping, log_evaluation


# # # === Set Random Seed for Reproducibility ===
# # np.random.seed(333)
# # tf.random.set_seed(333)

# # # === Set Paths ===
# # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # base_path = f'./_save/dacon_cancer/{timestamp}'
# # os.makedirs(base_path, exist_ok=True)
# # data_path = './_data/dacon/cancer/'

# # # === Load Data ===
# # train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
# # test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
# # submission_csv = pd.read_csv(data_path + 'sample_submission.csv', index_col=0)

# # # === Preprocessing (as before) ===
# # x = train_csv.drop(['Cancer'], axis=1)
# # y = train_csv['Cancer']
# # x = pd.get_dummies(x, drop_first=True)
# # test_csv = pd.get_dummies(test_csv, drop_first=True)
# # x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)

# # scaler = MinMaxScaler()
# # x_scaled = scaler.fit_transform(x)
# # test_scaled = scaler.transform(test_csv)

# # xgb_fs = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
# # xgb_fs.fit(x_scaled, y)
# # importances = xgb_fs.feature_importances_
# # threshold = np.percentile(importances, 25)
# # selected_indices = np.where(importances > threshold)[0]
# # x_selected = x_scaled[:, selected_indices]
# # test_selected = test_scaled[:, selected_indices]

# # # === Train/Validation Split ===
# # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# # for train_idx, val_idx in sss.split(x_selected, y):
# #     x_train, x_val = x_selected[train_idx], x_selected[val_idx]
# #     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# # # --- 💡 IMPROVEMENT 1: Calculate Class Weights ---
# # # Calculate weights to handle class imbalance without SMOTE.
# # neg_count, pos_count = y_train.value_counts().sort_index()
# # scale_pos_weight_value = neg_count / pos_count
# # class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# # class_weights_dict = {i : class_weights[i] for i in range(len(class_weights))}
# # print(f"✅ Scale Pos Weight for Boosting: {scale_pos_weight_value:.2f}")
# # print(f"✅ Class Weights for Keras: {class_weights_dict}")


# # # === Neural Network Model Builder ===
# # # === Neural Network Model Builder (Corrected Version) ===
# # def build_model(input_dim):
# #     model = Sequential([
# #         Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.001)),
# #         BatchNormalization(), 
# #         Dropout(0.2),
        
# #         Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
# #         BatchNormalization(), 
# #         Dropout(0.3),
        
# #         Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
# #         BatchNormalization(), 
# #         Dropout(0.2),
        
# #         Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
# #         BatchNormalization(), 
# #         Dropout(0.2),
        
# #         Dense(1, activation='sigmoid')
# #     ])
    
# #     # metrics에 Precision과 Recall을 다시 추가하는 것이 F1 스코어 모니터링에 좋습니다.
# #     model.compile(loss='binary_crossentropy', optimizer='adam',
# #                   metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')])
# #     return model

# # # === Callbacks (shared) ===
# # es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
# # lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0)

# # # === Train Neural Network Ensemble ===
# # n_models = 3
# # val_preds_nn = []
# # nn_test_preds = []
# # models = []

# # for i in range(n_models):
# #     print(f"\n🔁 Training NN model {i+1}/{n_models}")
# #     model = build_model(x_train.shape[1])
# #     model_path = os.path.join(base_path, f"nn_model_{i}.h5")
# #     mcp = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=0)
    
# #     # Apply class_weight here
# #     model.fit(x_train, y_train,
# #               validation_data=(x_val, y_val),
# #               epochs=30, batch_size=16,
# #               callbacks=[es, lr, mcp],
# #               class_weight=class_weights_dict, # 💡 Apply class weights
# #               verbose=1)
    
# #     best_model = load_model(model_path)
# #     models.append(best_model)
# #     val_preds_nn.append(best_model.predict(x_val).ravel())
# #     nn_test_preds.append(best_model.predict(test_selected).ravel())

# # val_pred_nn_avg = np.mean(val_preds_nn, axis=0)
# # test_pred_nn_avg = np.mean(nn_test_preds, axis=0)

# # # === Train Boosting Models with Class Weights ===
# # print("\n⚡ Training XGBoost with class weights...")
# # xgb_model = XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight_value) # 💡 Apply weight
# # xgb_model.fit(x_train, y_train)
# # xgb_val_pred = xgb_model.predict_proba(x_val)[:, 1]
# # xgb_test_pred = xgb_model.predict_proba(test_selected)[:, 1]

# # # --- 💡 IMPROVEMENT 2: Hyperparameter Tuning (Example with Optuna for LightGBM) ---
# # def lgb_objective(trial):
# #     params = {
# #         'objective': 'binary',
# #         'metric': 'binary_logloss',
# #         'n_estimators': 1000,
# #         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
# #         'num_leaves': trial.suggest_int('num_leaves', 20, 300),
# #         'max_depth': trial.suggest_int('max_depth', 3, 10),
# #         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
# #         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
# #         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
# #         'random_state': 42,
# #         'n_jobs': -1,
# #         'scale_pos_weight': scale_pos_weight_value # Apply weight
# #     }
    
# #     model = lgb.LGBMClassifier(**params)
# #     model.fit(x_train, y_train, eval_set=[(x_val, y_val)], callbacks=[early_stopping(30), log_evaluation(0)])
# #     preds = model.predict_proba(x_val)[:, 1]
# #     return roc_auc_score(y_val, preds) # Optimize for AUC, as it's less threshold-dependent

# # print("\n⚙️ Tuning LightGBM with Optuna...")
# # study = optuna.create_study(direction='maximize')
# # study.optimize(lgb_objective, n_trials=50) # Increase n_trials for better results
# # lgb_best_params = study.best_params
# # print("🏆 Best LightGBM Params:", lgb_best_params)

# # print("⚡ Training LightGBM with best params...")
# # lgb_model = lgb.LGBMClassifier(**lgb_best_params, n_estimators=1000, random_state=42, scale_pos_weight=scale_pos_weight_value)
# # lgb_model.fit(x_train, y_train, eval_set=[(x_val, y_val)], callbacks=[early_stopping(30), log_evaluation(0)])
# # lgb_val_pred = lgb_model.predict_proba(x_val)[:, 1]
# # lgb_test_pred = lgb_model.predict_proba(test_selected)[:, 1]


# # print("⚡ Training CatBoost with class weights...")
# # cat_model = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, verbose=0, random_state=42,
# #                                scale_pos_weight=scale_pos_weight_value) # 💡 Apply weight
# # cat_model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=30)
# # cat_val_pred = cat_model.predict_proba(x_val)[:, 1]
# # cat_test_pred = cat_model.predict_proba(test_selected)[:, 1]

# # # === Meta-Model Stacking ===
# # print("\n🔗 Stacking models...")
# # stacked_val = np.column_stack([val_pred_nn_avg, xgb_val_pred, lgb_val_pred, cat_val_pred])
# # meta_model = LogisticRegression()
# # meta_model.fit(stacked_val, y_val)

# # # Predict probabilities, not classes
# # stacked_val_proba = meta_model.predict_proba(stacked_val)[:, 1]

# # # --- 💡 IMPROVEMENT 3: Find Best Threshold for Final Prediction ---
# # print("🔍 Finding best F1 threshold for stacked model...")
# # thresholds = np.arange(0.1, 0.9, 0.01)
# # f1_scores = [f1_score(y_val, (stacked_val_proba > t).astype(int)) for t in thresholds]
# # best_idx = np.argmax(f1_scores)
# # best_threshold = thresholds[best_idx]
# # best_f1 = f1_scores[best_idx]

# # print(f"🏆 Final Stacked F1 on Validation Set: {best_f1:.4f} at threshold {best_threshold:.2f}")

# # # === Predict Test Data and Apply Best Threshold ===
# # print("\n🚀 Predicting on test set with final stacked model...")
# # stacked_test = np.column_stack([test_pred_nn_avg, xgb_test_pred, lgb_test_pred, cat_test_pred])
# # final_test_proba = meta_model.predict_proba(stacked_test)[:, 1]

# # # Apply the best threshold found on the validation set
# # final_test_pred = (final_test_proba > best_threshold).astype(int)

# # # === Save Submission ===
# # submission_csv['Cancer'] = final_test_pred
# # submission_path = os.path.join(base_path, f'submission_final_{timestamp}.csv')
# # submission_csv.to_csv(submission_path)
# # print(f"✅ Submission saved: {submission_path}")



# # === Import Libraries ===
# import pandas as pd
# import numpy as np
# import os
# import time
# from datetime import datetime
# import optuna

# # Keras and TensorFlow
# import tensorflow as tf
# from keras.models import Sequential, load_model
# from keras.layers import Dense, Dropout, BatchNormalization
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from keras.regularizers import l2
# from keras.metrics import AUC, Precision, Recall

# # Scikit-learn and Boosting
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import f1_score, roc_auc_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.utils.class_weight import compute_class_weight
# from xgboost import XGBClassifier
# import lightgbm as lgb
# from catboost import CatBoostClassifier
# from lightgbm import early_stopping, log_evaluation

# # === Set Random Seed for Reproducibility ===
# np.random.seed(333)
# tf.random.set_seed(333)

# # === Set Paths ===
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# base_path = f'./_save/dacon_cancer/{timestamp}'
# os.makedirs(base_path, exist_ok=True)
# data_path = './_data/dacon/cancer/'

# # === Load Data ===
# train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
# test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
# submission_csv = pd.read_csv(data_path + 'sample_submission.csv', index_col=0)

# # === Preprocessing ===
# print("1. Starting data preprocessing...")
# x = train_csv.drop(['Cancer'], axis=1)
# y = train_csv['Cancer']
# x = pd.get_dummies(x, drop_first=True)
# test_csv = pd.get_dummies(test_csv, drop_first=True)
# x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)

# scaler = MinMaxScaler()
# x_scaled = scaler.fit_transform(x)
# test_scaled = scaler.transform(test_csv)

# print("2. Performing feature selection with XGBoost...")
# xgb_fs = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
# xgb_fs.fit(x_scaled, y)
# importances = xgb_fs.feature_importances_
# threshold = np.percentile(importances, 5)
# selected_indices = np.where(importances > threshold)[0]
# x_selected = x_scaled[:, selected_indices]
# test_selected = test_scaled[:, selected_indices]
# print(f"   Selected {x_selected.shape[1]} features from original {x_scaled.shape[1]}.")

# # === Train/Validation Split ===
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_idx, val_idx in sss.split(x_selected, y):
#     x_train, x_val = x_selected[train_idx], x_selected[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# # --- Calculate Class Weights ---
# neg_count, pos_count = y_train.value_counts().sort_index()
# scale_pos_weight_value = neg_count / pos_count
# class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# class_weights_dict = {i : class_weights[i] for i in range(len(class_weights))}
# print(f"\n✅ Scale Pos Weight for Boosting: {scale_pos_weight_value:.2f}")
# print(f"✅ Class Weights for Keras: {class_weights_dict}")

# # === Neural Network Model Builder ===
# def build_model(input_dim):
#     model = Sequential([
#         Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(), Dropout(0.2),
#         Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(), Dropout(0.3),
#         Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(), Dropout(0.2),
#         Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(), Dropout(0.2),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(loss='binary_crossentropy', optimizer='adam',
#                   metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')])
#     return model

# # === Callbacks (shared) ===
# es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
# lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0)

# # === Train Neural Network Ensemble ===
# print("\n3. Training Level-0 Models...")
# n_models = 3
# val_preds_nn = []
# nn_test_preds = []

# for i in range(n_models):
#     print(f"   🔁 Training NN model {i+1}/{n_models}")
#     model = build_model(x_train.shape[1])
#     model_path = os.path.join(base_path, f"nn_model_{i}.h5")
#     mcp = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=0)
    
#     model.fit(x_train, y_train,
#               validation_data=(x_val, y_val),
#               epochs=70, batch_size=32,
#               callbacks=[es, lr, mcp],
#               class_weight=class_weights_dict,
#               verbose=1)
    
#     best_model = load_model(model_path)
#     val_preds_nn.append(best_model.predict(x_val).ravel())
#     nn_test_preds.append(best_model.predict(test_selected).ravel())

# val_pred_nn_avg = np.mean(val_preds_nn, axis=0)
# test_pred_nn_avg = np.mean(nn_test_preds, axis=0)

# # === Train Boosting Models with Class Weights ===
# print("   ⚡ Training XGBoost with class weights...")
# # 올바른 코드
# print("   ⚡ Training XGBoost with class weights...")
# xgb_model = XGBClassifier(
#     eval_metric='logloss', 
#     random_state=42, 
#     scale_pos_weight=scale_pos_weight_value, 
#     n_estimators=2000,
#     early_stopping_rounds=50  # 파라미터를 이 위치로 이동
# )
# # fit에서는 eval_set과 verbose만 남김
# xgb_model.fit(
#     x_train, y_train, 
#     eval_set=[(x_val, y_val)], 
#     verbose=False
# )
# xgb_val_pred = xgb_model.predict_proba(x_val)[:, 1]
# xgb_test_pred = xgb_model.predict_proba(test_selected)[:, 1]

# # --- Optuna Objective Function for LightGBM ---
# def lgb_objective(trial):
#     params = {
#         'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
#         'boosting_type': 'gbdt', 'n_estimators': 1000,
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 150),
#         'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.05),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
#         'random_state': 42, 'n_jobs': -1,
#         'scale_pos_weight': scale_pos_weight_value
#     }
#     model = lgb.LGBMClassifier(**params)
#     model.fit(x_train, y_train, eval_set=[(x_val, y_val)], callbacks=[early_stopping(50, verbose=False)])
#     preds = model.predict_proba(x_val)[:, 1]
#     return roc_auc_score(y_val, preds)

# print("   ⚙️ Tuning LightGBM with Optuna...")
# study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
# study.optimize(lgb_objective, n_trials=50, show_progress_bar=False) # Set show_progress_bar to True for details
# lgb_best_params = study.best_params
# print("   🏆 Best LightGBM Params found.")

# print("   ⚡ Training LightGBM with best params...")
# lgb_model = lgb.LGBMClassifier(**lgb_best_params, n_estimators=2000, random_state=42)
# lgb_model.fit(x_train, y_train, eval_set=[(x_val, y_val)], callbacks=[early_stopping(50, verbose=False)])
# lgb_val_pred = lgb_model.predict_proba(x_val)[:, 1]
# lgb_test_pred = lgb_model.predict_proba(test_selected)[:, 1]

# print("   ⚡ Training CatBoost with class weights...")
# cat_model = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, verbose=0, random_state=42,
#                                scale_pos_weight=scale_pos_weight_value)
# cat_model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=50)
# cat_val_pred = cat_model.predict_proba(x_val)[:, 1]
# cat_test_pred = cat_model.predict_proba(test_selected)[:, 1]

# # === Meta-Model Stacking ===
# print("\n4. Training Level-1 Meta-Model (Stacking)...")
# stacked_val = np.column_stack([val_pred_nn_avg, xgb_val_pred, lgb_val_pred, cat_val_pred])
# meta_model = LogisticRegression(random_state=42)
# meta_model.fit(stacked_val, y_val)
# stacked_val_proba = meta_model.predict_proba(stacked_val)[:, 1]

# # === Find Best Threshold for Final Prediction ===
# print("\n5. Finding best F1 threshold...")
# thresholds = np.arange(0.1, 0.9, 0.01)
# f1_scores = [f1_score(y_val, (stacked_val_proba > t).astype(int)) for t in thresholds]
# best_idx = np.argmax(f1_scores)
# best_threshold = thresholds[best_idx]
# best_f1 = f1_scores[best_idx]
# print(f"   🏆 Final Stacked F1 on Validation Set: {best_f1:.4f} at threshold {best_threshold:.2f}")

# # === Predict Test Data and Apply Best Threshold ===
# print("\n6. Generating final predictions on test set...")
# stacked_test = np.column_stack([test_pred_nn_avg, xgb_test_pred, lgb_test_pred, cat_test_pred])
# final_test_proba = meta_model.predict_proba(stacked_test)[:, 1]
# final_test_pred = (final_test_proba > best_threshold).astype(int)

# # === Save Submission ===
# submission_csv['Cancer'] = final_test_pred
# submission_path = os.path.join(base_path, f'submission_final_f1_{best_f1:.4f}_{timestamp}.csv')
# submission_csv.to_csv(submission_path)
# print(f"\n✅ Submission saved: {submission_path}")



# === Import Libraries ===
import pandas as pd
import numpy as np
import os
from datetime import datetime
import optuna
from category_encoders import CatBoostEncoder

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2
from keras.metrics import AUC, Precision, Recall
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from lightgbm import early_stopping

# === Random Seed ===
np.random.seed(42)
tf.random.set_seed(42)

# === Path Setup ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_path = f'./_save/dacon_cancer/{timestamp}'
os.makedirs(base_path, exist_ok=True)
data_path = '/Users/jaewoo000/Desktop/IBM:Redhat/Study25/_data/dacon/cancer/'

# === Load Data ===
train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(data_path + 'sample_submission.csv', index_col=0)
def preprocess_final(train_df, test_df):
    y = train_df['Cancer'].reset_index(drop=True)
    train_df = train_df.drop(columns='Cancer')

    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # 파생 변수
    full_df['Hormone_Sum'] = full_df['TSH_Result'] + full_df['T3_Result'] + full_df['T4_Result']
    full_df['TSH_T3_ratio'] = full_df['TSH_Result'] / (full_df['T3_Result'] + 1e-3)
    full_df['T3_T4_ratio'] = full_df['T3_Result'] / (full_df['T4_Result'] + 1e-3)

    # 분할
    x_train_raw = full_df.iloc[:len(train_df)].copy().reset_index(drop=True)
    x_test_raw = full_df.iloc[len(train_df):].copy().reset_index(drop=True)

    # Categorical 인코딩
    cat_cols = ['Gender', 'Country', 'Race', 'Family_Background',
                'Radiation_History', 'Iodine_Deficiency', 'Smoke',
                'Weight_Risk', 'Diabetes']
    
    from category_encoders import CatBoostEncoder
    cbe = CatBoostEncoder(cols=cat_cols)
    x_train_raw[cat_cols] = cbe.fit_transform(x_train_raw[cat_cols], y)
    x_test_raw[cat_cols] = cbe.transform(x_test_raw[cat_cols])

    # 스케일링
    from sklearn.preprocessing import RobustScaler
    num_cols = x_train_raw.select_dtypes(include=np.number).columns
    scaler = RobustScaler()
    x_train_raw[num_cols] = scaler.fit_transform(x_train_raw[num_cols])
    x_test_raw[num_cols] = scaler.transform(x_test_raw[num_cols])

    return x_train_raw, y, x_test_raw



# === Preprocessing ===
print("1. Starting data preprocessing...")
x, y, test_csv = preprocess_final(train_csv, test_csv)



# === Feature Selection with XGBoost ===
print("2. Performing feature selection with XGBoost...")
xgb_fs = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_fs.fit(x, y)
# 🔽 바로 여기부터 붙이세요
import matplotlib.pyplot as plt
from xgboost import plot_importance

plt.figure(figsize=(12, 8))
plot_importance(xgb_fs, max_num_features=30, importance_type='gain')  # gain 기준 상위 30개
plt.title("XGBoost Feature Importance (Top 30 by Gain)")
plt.tight_layout()

# 저장 경로 지정
importance_path = os.path.join(base_path, f'xgb_feature_importance_{timestamp}.png')
plt.savefig(importance_path)
plt.show()
print(f"✅ Feature importance plot saved to {importance_path}")


importances = xgb_fs.feature_importances_
threshold = np.percentile(importances, 1)
selected_indices = np.where(importances > threshold)[0]
x_selected = x.iloc[:, selected_indices]
test_selected = test_csv.iloc[:, selected_indices]
print(f"   Selected {x_selected.shape[1]} features from original {x.shape[1]}.")

# === Train/Val Split ===
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in sss.split(x_selected, y):
    x_train, x_val = x_selected.iloc[train_idx], x_selected.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# === Class Weights ===
neg_count, pos_count = y_train.value_counts().sort_index()
scale_pos_weight_value = neg_count / pos_count
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"\n✅ Scale Pos Weight for Boosting: {scale_pos_weight_value:.2f}")
print(f"✅ Class Weights for Keras: {class_weights_dict}")

# === Build NN Model ===
def build_model(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(), Dropout(0.2),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(), Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(), Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(), Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')])
    return model

# === Callbacks ===
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0)

# === Train Neural Networks ===
print("\n3. Training Level-0 Models...")
n_models = 3
val_preds_nn = []
nn_test_preds = []

for i in range(n_models):
    print(f"   🔁 Training NN model {i+1}/{n_models}")
    model = build_model(x_train.shape[1])
    model_path = os.path.join(base_path, f"nn_model_{i}.h5")
    mcp = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=0)

    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              epochs=15, batch_size=32,
              callbacks=[es, lr, mcp],
              class_weight=class_weights_dict,
              verbose=1)
    
    best_model = load_model(model_path)
    val_preds_nn.append(best_model.predict(x_val).ravel())
    nn_test_preds.append(best_model.predict(test_selected).ravel())

val_pred_nn_avg = np.mean(val_preds_nn, axis=0)
test_pred_nn_avg = np.mean(nn_test_preds, axis=0)

# === XGBoost ===
print("   ⚡ Training XGBoost with class weights...")
xgb_model = XGBClassifier(
    eval_metric='logloss', 
    random_state=42, 
    scale_pos_weight=scale_pos_weight_value, 
    n_estimators=2000,
    early_stopping_rounds=50
)
xgb_model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

# 🔽 여기다 SHAP 코드 삽입
# import shap
# print("\n🔍 Calculating SHAP values...")
# explainer = shap.TreeExplainer(xgb_model)
# shap_values = explainer.shap_values(x_train)

# plt.figure(figsize=(12, 8))
# shap.summary_plot(shap_values, x_train, plot_type="bar")
# shap.summary_plot(shap_values, x_train)
# 🔼 여기까지

xgb_val_pred = xgb_model.predict_proba(x_val)[:, 1]
xgb_test_pred = xgb_model.predict_proba(test_selected)[:, 1]
xgb_val_pred = xgb_model.predict_proba(x_val)[:, 1]
xgb_test_pred = xgb_model.predict_proba(test_selected)[:, 1]

# === LightGBM Optuna Tuning ===
def lgb_objective(trial):
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
        'boosting_type': 'gbdt', 'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.05),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': 42, 'n_jobs': -1,
        'scale_pos_weight': scale_pos_weight_value
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], callbacks=[early_stopping(50, verbose=False)])
    preds = model.predict_proba(x_val)[:, 1]
    return roc_auc_score(y_val, preds)

print("   ⚙️ Tuning LightGBM with Optuna...")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lgb_objective, n_trials=50, show_progress_bar=False)
lgb_best_params = study.best_params
print("   🏆 Best LightGBM Params found.")

print("   ⚡ Training LightGBM with best params...")
lgb_model = lgb.LGBMClassifier(**lgb_best_params, n_estimators=2000, random_state=42)
lgb_model.fit(x_train, y_train, eval_set=[(x_val, y_val)], callbacks=[early_stopping(50, verbose=False)])
lgb_val_pred = lgb_model.predict_proba(x_val)[:, 1]
lgb_test_pred = lgb_model.predict_proba(test_selected)[:, 1]

# === CatBoost ===
print("   ⚡ Training CatBoost with class weights...")
cat_model = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, verbose=0, random_state=42,
                               scale_pos_weight=scale_pos_weight_value)
cat_model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=50)
cat_val_pred = cat_model.predict_proba(x_val)[:, 1]
cat_test_pred = cat_model.predict_proba(test_selected)[:, 1]

# === Meta Model ===
print("\n4. Training Level-1 Meta-Model (Stacking)...")
stacked_val = np.column_stack([val_pred_nn_avg, xgb_val_pred, lgb_val_pred, cat_val_pred])
meta_model = LogisticRegression(random_state=42)
meta_model.fit(stacked_val, y_val)
stacked_val_proba = meta_model.predict_proba(stacked_val)[:, 1]

# === Optimal Threshold ===
print("\n5. Finding best F1 threshold...")
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_val, (stacked_val_proba > t).astype(int)) for t in thresholds]
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
print(f"   🏆 Final Stacked F1 on Validation Set: {best_f1:.4f} at threshold {best_threshold:.2f}")

# === Final Prediction ===
print("\n6. Generating final predictions on test set...")
stacked_test = np.column_stack([test_pred_nn_avg, xgb_test_pred, lgb_test_pred, cat_test_pred])
final_test_proba = meta_model.predict_proba(stacked_test)[:, 1]
final_test_pred = (final_test_proba > best_threshold).astype(int)

submission_csv['Cancer'] = final_test_pred
submission_path = os.path.join(base_path, f'submission_final_f1_{best_f1:.4f}_{timestamp}.csv')
submission_csv.to_csv(submission_path)
print(f"\n✅ Submission saved: {submission_path}")


