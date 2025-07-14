# # # # === Import Libraries ===
# # # import pandas as pd
# # # import numpy as np
# # # import os
# # # from datetime import datetime
# # # from imblearn.over_sampling import SMOTE

# # # from keras.models import Sequential
# # # from keras.layers import Dense, Dropout, BatchNormalization
# # # from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# # # from keras.regularizers import l2
# # # from keras.metrics import AUC, Precision, Recall

# # # from sklearn.model_selection import StratifiedShuffleSplit
# # # from sklearn.preprocessing import MinMaxScaler
# # # from sklearn.metrics import f1_score

# # # # === Set Random Seed for Reproducibility ===
# # # np.random.seed(42)

# # # # === Set Time-based Save Path ===
# # # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # base_path = f'/Users/jaewoo/Desktop/IBM x RedHat/Study25/_save/dacon_cancer/{timestamp}'
# # # os.makedirs(base_path, exist_ok=True)

# # # # === Load Data ===
# # # data_path = '/Users/jaewoo/Desktop/IBM x RedHat/Study25/_data/dacon/cancer/'
# # # train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
# # # test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
# # # submission_csv = pd.read_csv(data_path + 'sample_submission.csv', index_col=0)

# # # # === Separate Features and Target Label ===
# # # x = train_csv.drop(['Cancer'], axis=1)
# # # y = train_csv['Cancer']

# # # # === One-Hot Encode Categorical Columns ===
# # # categorical_cols = x.select_dtypes(include='object').columns
# # # x = pd.get_dummies(x, columns=categorical_cols)
# # # test_csv = pd.get_dummies(test_csv, columns=categorical_cols)
# # # x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)

# # # # === Scale Data ===
# # # scaler = MinMaxScaler()
# # # x = scaler.fit_transform(x)
# # # test_csv = scaler.transform(test_csv)

# # # # === Stratified Train/Validation Split ===
# # # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# # # for train_idx, val_idx in sss.split(x, y):
# # #     x_train, x_val = x[train_idx], x[val_idx]
# # #     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# # # # === Apply SMOTE only to training set ===
# # # x_train, y_train = SMOTE(random_state=333).fit_resample(x_train, y_train)

# # # # === Build the Model ===
# # # model = Sequential([
# # #     Dense(128, input_dim=x.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
# # #     BatchNormalization(),
# # #     Dropout(0.2),
    
# # #     Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
# # #     BatchNormalization(),
# # #     Dropout(0.3),
    
# # #     Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
# # #     BatchNormalization(),
# # #     Dropout(0.2),
    
# # #     Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
# # #     BatchNormalization(),
# # #     Dropout(0.2),
    
# # #     Dense(1, activation='sigmoid')
# # # ])

# # # # === Compile Model ===
# # # model.compile(
# # #     loss='binary_crossentropy',
# # #     optimizer='adam',
# # #     metrics=['accuracy', AUC(name='auc'), Precision(), Recall()]
# # # )

# # # # === Callbacks ===
# # # model_path = os.path.join(base_path, f'model_mcp_{timestamp}.h5')
# # # mcp = ModelCheckpoint(
# # #     model_path,
# # #     monitor='val_loss',
# # #     save_best_only=True,
# # #     verbose=1
# # # )
# # # es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
# # # lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

# # # # === Train Model ===
# # # model.fit(
# # #     x_train, y_train,
# # #     validation_data=(x_val, y_val),
# # #     epochs=1000,
# # #     batch_size=32,
# # #     callbacks=[es, lr, mcp],
# # #     verbose=1
# # # )

# # # # === Evaluate Model ===
# # # loss, accuracy, auc, precision, recall = model.evaluate(x_val, y_val, verbose=0)
# # # y_val_pred = model.predict(x_val).ravel()

# # # # === Find Best Threshold for F1 Score ===
# # # thresholds = np.arange(0.3, 0.7, 0.01)
# # # f1_scores = [f1_score(y_val, (y_val_pred > t).astype(int)) for t in thresholds]
# # # best_idx = np.argmax(f1_scores)
# # # best_threshold = thresholds[best_idx]
# # # best_f1 = f1_scores[best_idx]

# # # # === Print Metrics ===
# # # print(f'✅ loss: {loss:.4f}')
# # # print(f'✅ acc : {accuracy:.4f}')
# # # print(f'✅ AUC : {auc:.4f}')
# # # print(f'✅ Precision: {precision:.4f}')
# # # print(f'✅ Recall   : {recall:.4f}')
# # # print(f'✅ Best F1: {best_f1:.4f} at threshold {best_threshold:.2f}')

# # # # === Predict on Final Test Set and Save Submission ===
# # # y_submit = model.predict(test_csv).ravel()
# # # submission_csv['Cancer'] = (y_submit > best_threshold).astype(int)
# # # submission_path = os.path.join(base_path, f'submission_{timestamp}.csv')
# # # submission_csv.to_csv(submission_path)
# # # print(f"✅ Submission saved to: {submission_path}")
# # # print(f"✅ Best model saved to: {model_path}")

# ######best verion#############best verion#############best verion#######
# ######best verion#######
# ######best verion####### below!!
# # === Import Libraries ===
# import pandas as pd
# import numpy as np
# import os
# from datetime import datetime
# from imblearn.over_sampling import SMOTE

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, BatchNormalization
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from keras.regularizers import l2
# from keras.metrics import AUC, Precision, Recall

# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import f1_score

# from xgboost import XGBClassifier

# # === Set Random Seed for Reproducibility ===
# np.random.seed(333)

# # === Set Time-based Save Path ===
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# base_path = f'/Users/jaewoo/Desktop/IBM x RedHat/Study25/_save/dacon_cancer/{timestamp}'
# os.makedirs(base_path, exist_ok=True)

# # === Load Data ===
# data_path = '/Users/jaewoo/Desktop/IBM x RedHat/Study25/_data/dacon/cancer/'
# train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
# test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
# submission_csv = pd.read_csv(data_path + 'sample_submission.csv', index_col=0)

# # === Separate Features and Target Label ===
# x = train_csv.drop(['Cancer'], axis=1)
# y = train_csv['Cancer']

# # === One-Hot Encode Categorical Columns ===
# categorical_cols = x.select_dtypes(include='object').columns
# x = pd.get_dummies(x, columns=categorical_cols)
# test_csv = pd.get_dummies(test_csv, columns=categorical_cols)
# x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)

# # === Scale Data ===
# scaler = MinMaxScaler()
# x_scaled = scaler.fit_transform(x)
# test_scaled = scaler.transform(test_csv)

# # === Feature Selection with XGBoost ===
# xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
# xgb.fit(x_scaled, y)
# importances = xgb.feature_importances_
# threshold = np.percentile(importances, 25)  # remove bottom 25% features
# selected_indices = np.where(importances > threshold)[0]
# x_selected = x_scaled[:, selected_indices]
# test_selected = test_scaled[:, selected_indices]

# # === Stratified Train/Validation Split ===
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_idx, val_idx in sss.split(x_selected, y):
#     x_train, x_val = x_selected[train_idx], x_selected[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# # === Apply SMOTE only to training set ===
# x_train, y_train = SMOTE(random_state=333).fit_resample(x_train, y_train)

# # === Build the Model ===
# model = Sequential([
#     Dense(128, input_dim=x_selected.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
#     BatchNormalization(),
#     Dropout(0.2),
#     Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
#     BatchNormalization(),
#     Dropout(0.3),
#     Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
#     BatchNormalization(),
#     Dropout(0.2),
#     Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
#     BatchNormalization(),
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')
# ])

# # === Compile Model ===
# model.compile(
#     loss='binary_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy', AUC(name='auc'), Precision(), Recall()]
# )

# # === Callbacks ===
# model_path = os.path.join(base_path, f'model_mcp_{timestamp}.h5')
# mcp = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
# es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
# lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

# # === Train Model ===
# model.fit(
#     x_train, y_train,
#     validation_data=(x_val, y_val),
#     epochs=1000,
#     batch_size=16,
#     callbacks=[es, lr, mcp],
#     verbose=1
# )

# # === Evaluate Model ===
# loss, accuracy, auc, precision, recall = model.evaluate(x_val, y_val, verbose=0)
# y_val_pred = model.predict(x_val).ravel()

# # === Find Best Threshold for F1 Score ===
# thresholds = np.arange(0.2, 0.8, 0.01)
# f1_scores = [f1_score(y_val, (y_val_pred > t).astype(int)) for t in thresholds]
# best_idx = np.argmax(f1_scores)
# best_threshold = thresholds[best_idx]
# best_f1 = f1_scores[best_idx]

# # === Print Metrics ===
# print(f'✅ loss: {loss:.4f}')
# print(f'✅ acc : {accuracy:.4f}')
# print(f'✅ AUC : {auc:.4f}')
# print(f'✅ Precision: {precision:.4f}')
# print(f'✅ Recall   : {recall:.4f}')
# print(f'✅ Best F1: {best_f1:.4f} at threshold {best_threshold:.2f}')

# # === Predict on Final Test Set and Save Submission ===
# y_submit = model.predict(test_selected).ravel()
# submission_csv['Cancer'] = (y_submit > best_threshold).astype(int)
# submission_path = os.path.join(base_path, f'submission_{timestamp}.csv')
# submission_csv.to_csv(submission_path)
# print(f"✅ Submission saved to: {submission_path}")
# print(f"✅ Best model saved to: {model_path}")




# ###########
# ###########
# ###########
# ###########
# ###########


# # # === Import Libraries ===
# # import pandas as pd
# # import numpy as np
# # import os
# # from datetime import datetime
# # from imblearn.over_sampling import SMOTE
# # from xgboost import XGBClassifier
# # from sklearn.feature_selection import SelectFromModel
# # from sklearn.model_selection import StratifiedShuffleSplit
# # from sklearn.preprocessing import MinMaxScaler
# # from sklearn.metrics import f1_score
# # from sklearn.utils.class_weight import compute_class_weight
# # import matplotlib.pyplot as plt

# # from keras.models import Sequential
# # from keras.layers import Dense, Dropout, BatchNormalization
# # from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# # from keras.regularizers import l2
# # from keras.metrics import AUC, Precision, Recall

# # # === Reproducibility ===
# # np.random.seed(42)

# # # === Paths ===
# # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # base_path = f'/Users/jaewoo/Desktop/IBM x RedHat/Study25/_save/dacon_cancer/{timestamp}'
# # os.makedirs(base_path, exist_ok=True)
# # data_path = '/Users/jaewoo/Desktop/IBM x RedHat/Study25/_data/dacon/cancer/'

# # # === Load Data ===
# # train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
# # test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
# # submission_csv = pd.read_csv(data_path + 'sample_submission.csv', index_col=0)

# # x = train_csv.drop(['Cancer'], axis=1)
# # y = train_csv['Cancer']

# # # === One-Hot Encode ===
# # categorical_cols = x.select_dtypes(include='object').columns
# # x = pd.get_dummies(x, columns=categorical_cols)
# # test_csv = pd.get_dummies(test_csv, columns=categorical_cols)
# # x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)

# # # === Scale ===
# # scaler = MinMaxScaler()
# # x = scaler.fit_transform(x)
# # test_csv = scaler.transform(test_csv)

# # # === SMOTE ===
# # x, y = SMOTE(random_state=42).fit_resample(x, y)

# # # === Feature Selection using XGBoost + SelectFromModel ===
# # xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
# # xgb.fit(x, y)
# # selector = SelectFromModel(xgb, threshold='mean', prefit=True)
# # x_selected = selector.transform(x)
# # test_selected = selector.transform(test_csv)

# # # === Train/Val Split ===
# # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# # for train_idx, val_idx in sss.split(x_selected, y):
# #     x_train, x_val = x_selected[train_idx], x_selected[val_idx]
# #     y_train, y_val = y[train_idx], y[val_idx]

# # # === Class Weight ===
# # class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# # class_weights_dict = dict(enumerate(class_weights))

# # # === Build Model (간소화 구조) ===
# # model = Sequential([
# #     Dense(128, input_dim=x_selected.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
# #     BatchNormalization(),
# #     Dropout(0.2),
# #     Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
# #     BatchNormalization(),
# #     Dropout(0.2),
# #     Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
# #     BatchNormalization(),
# #     Dropout(0.2),
# #     Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
# #     BatchNormalization(),
# #     Dropout(0.2),
# #     Dense(1, activation='sigmoid')
# # ])

# # model.compile(
# #     loss='binary_crossentropy',
# #     optimizer='adam',
# #     metrics=['accuracy', AUC(name='auc'), Precision(), Recall()]
# # )

# # # === Callbacks ===
# # model_path = os.path.join(base_path, f'model_{timestamp}.h5')
# # callbacks = [
# #     ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1),
# #     EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
# #     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
# # ]

# # # === Train ===
# # model.fit(
# #     x_train, y_train,
# #     validation_data=(x_val, y_val),
# #     epochs=1000,
# #     batch_size=32,
# #     class_weight=class_weights_dict,
# #     callbacks=callbacks,
# #     verbose=1
# # )

# # # === Evaluate ===
# # loss, accuracy, auc, precision, recall = model.evaluate(x_val, y_val, verbose=0)
# # y_val_pred = model.predict(x_val).ravel()

# # # === Threshold Optimization (넓은 범위) ===
# # thresholds = np.arange(0.1, 0.9, 0.01)
# # f1_scores = [f1_score(y_val, (y_val_pred > t).astype(int)) for t in thresholds]
# # best_idx = np.argmax(f1_scores)
# # best_threshold = thresholds[best_idx]
# # best_f1 = f1_scores[best_idx]

# # # === 시각화 ===
# # plt.plot(thresholds, f1_scores)
# # plt.xlabel("Threshold")
# # plt.ylabel("F1 Score")
# # plt.title("Threshold vs F1")
# # plt.grid()
# # plt.savefig(os.path.join(base_path, f"threshold_f1_plot_{timestamp}.png"))

# # # === Print Results ===
# # print(f'✅ loss: {loss:.4f}')
# # print(f'✅ acc : {accuracy:.4f}')
# # print(f'✅ AUC : {auc:.4f}')
# # print(f'✅ Precision: {precision:.4f}')
# # print(f'✅ Recall   : {recall:.4f}')
# # print(f'✅ Best F1: {best_f1:.4f} at threshold {best_threshold:.2f}')

# # # === Final Submission ===
# # y_submit = model.predict(test_selected).ravel()
# # submission_csv['Cancer'] = (y_submit > best_threshold).astype(int)
# # submission_path = os.path.join(base_path, f'submission_{timestamp}.csv')
# # submission_csv.to_csv(submission_path)
# # print(f"✅ Submission saved to: {submission_path}")
# # print(f"✅ Best model saved to: {model_path}")



# # import pandas as pd
# # import numpy as np
# # import os
# # from datetime import datetime
# # from imblearn.over_sampling import SMOTE
# # from keras.models import Sequential
# # from keras.layers import Dense, Dropout, BatchNormalization
# # from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# # from keras.regularizers import l2
# # from keras.metrics import AUC, Precision, Recall
# # from sklearn.model_selection import StratifiedKFold
# # from sklearn.preprocessing import MinMaxScaler
# # from sklearn.metrics import f1_score
# # from sklearn.feature_selection import SelectFromModel
# # from xgboost import XGBClassifier

# # # 📌 Fix seed
# # np.random.seed(42)


# # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # base_path = f'/Users/jaewoo/Desktop/IBM x RedHat/Study25/_save/dacon_cancer/{timestamp}'
# # os.makedirs(base_path, exist_ok=True)

# # # === Load Data ===
# # data_path = '/Users/jaewoo/Desktop/IBM x RedHat/Study25/_data/dacon/cancer/'
# # train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
# # test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
# # submission_csv = pd.read_csv(data_path + 'sample_submission.csv', index_col=0)
# # # 📁 Paths
# # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # base_path = f'/Users/jaewoo/Desktop/IBM x RedHat/Study25/_save/dacon_cancer/{timestamp}'
# # os.makedirs(base_path, exist_ok=True)
# # data_path = '/Users/jaewoo/Desktop/IBM x RedHat/Study25/_data/dacon/cancer/'
# # train = pd.read_csv(data_path + 'train.csv', index_col=0)
# # test = pd.read_csv(data_path + 'test.csv', index_col=0)
# # submit = pd.read_csv(data_path + 'sample_submission.csv', index_col=0)

# # # 🔧 Features & Label
# # X = train.drop(columns=['Cancer'])
# # y = train['Cancer']
# # X = pd.get_dummies(X)
# # test = pd.get_dummies(test)
# # X, test = X.align(test, axis=1, fill_value=0)

# # # 🔄 Scale
# # scaler = MinMaxScaler()
# # X_scaled = scaler.fit_transform(X)
# # test_scaled = scaler.transform(test)

# # # 1️⃣ Feature selection via XGB + SelectFromModel
# # xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
# # xgb.fit(X_scaled, y)
# # sfm = SelectFromModel(xgb, threshold='median', prefit=True)
# # X_sel = sfm.transform(X_scaled)
# # test_sel = sfm.transform(test_scaled)

# # # 2️⃣ StratifiedKFold CV + ensemble preparatory
# # skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# # thresholds = np.arange(0.3, 0.9, 0.005)
# # y_val_true, y_val_pred = [], []

# # for fold, (trn_idx, val_idx) in enumerate(skf.split(X_sel, y), 1):
# #     X_tr, X_val = X_sel[trn_idx], X_sel[val_idx]
# #     y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

# #     # SMOTE on training only
# #     X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

# #     # Model
# #     model = Sequential([
# #         Dense(128, input_dim=X_sel.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
# #         BatchNormalization(),
# #         Dropout(0.2),
# #         Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
# #         BatchNormalization(),
# #         Dropout(0.3),
# #         Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
# #         BatchNormalization(),
# #         Dropout(0.2),
# #         Dense(1, activation='sigmoid'),
# #     ])
# #     model.compile(loss='binary_crossentropy', optimizer='adam',
# #                   metrics=['accuracy', AUC(name='auc'), Precision(), Recall()])

# #     # Callbacks
# #     ckpt = ModelCheckpoint(os.path.join(base_path, f'fold{fold}_best.h5'),
# #                            monitor='val_loss', save_best_only=True, verbose=0)
# #     es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0)
# #     rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0)

# #     model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
# #               epochs=500, batch_size=32,
# #               callbacks=[ckpt, es, rl], verbose=1)

# #     # Predict
# #     preds = model.predict(X_val).ravel()
# #     y_val_true.extend(y_val)
# #     y_val_pred.extend(preds)
# #     print(f"Fold {fold} done.")

# # # 3️⃣ Best threshold search
# # y_true = np.array(y_val_true)
# # y_pred = np.array(y_val_pred)
# # f1_scores = [f1_score(y_true, (y_pred > t).astype(int)) for t in thresholds]
# # best_idx = np.argmax(f1_scores)
# # best_t = thresholds[best_idx]
# # best_f1 = f1_scores[best_idx]

# # print(f"\n✅ Best Threshold: {best_t:.3f}, Best CV F1: {best_f1:.4f}")

# # # 🎯 Final model trained on full data
# # final = Sequential([
# #     Dense(128, input_dim=X_sel.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
# #     BatchNormalization(),
# #     Dropout(0.2),
# #     Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
# #     BatchNormalization(),
# #     Dropout(0.3),
# #     Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
# #     BatchNormalization(),
# #     Dropout(0.2),
# #     Dense(1, activation='sigmoid'),
# # ])
# # final.compile(loss='binary_crossentropy', optimizer='adam',
# #               metrics=['accuracy', AUC(name='auc'), Precision(), Recall()])

# # final.fit(X_sel, y, epochs=100, batch_size=32, verbose=1)

# # pred_test = final.predict(test_sel).ravel()
# # submit['Cancer'] = (pred_test > best_t).astype(int)
# # submit.to_csv(os.path.join(base_path, f'sub_{timestamp}.csv'))

# # print(f"✅ Submission saved at best threshold {best_t:.3f}")
