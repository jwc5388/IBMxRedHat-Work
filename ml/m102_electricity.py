# # # import os
# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.preprocessing import StandardScaler
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
# # # from sklearn.metrics import mean_absolute_error
# # # from xgboost import XGBRegressor
# # # from catboost import CatBoostRegressor
# # # from lightgbm import LGBMRegressor
# # # import random
# # # import tensorflow as tf

# # # # 고정 시드
# # # seed = 42
# # # random.seed(seed)
# # # np.random.seed(seed)
# # # tf.random.set_seed(seed)

# # # # 경로 설정
# # # if os.path.exists('/workspace/TensorJae/Study25/'):
# # #     BASE_PATH = '/workspace/TensorJae/Study25/'
# # # else:
# # #     BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')

# # # path = os.path.join(BASE_PATH, '_data/dacon/electricity/')
# # # save_path = os.path.join(path, '_save/')

# # # # 데이터 로드
# # # buildinginfo = pd.read_csv(path + 'building_info.csv')
# # # train = pd.read_csv(path + 'train.csv')
# # # test = pd.read_csv(path + 'test.csv')
# # # samplesubmission = pd.read_csv(path + 'sample_submission.csv')

# # # # 날짜 파싱 및 시간/요일 파생변수 추가
# # # for df in [train, test]:
# # #     df['일시'] = pd.to_datetime(df['일시'])
# # #     df['hour'] = df['일시'].dt.hour
# # #     df['dayofweek'] = df['일시'].dt.dayofweek

# # # # building_info의 '-' 값을 결측치로 처리 후 float 변환
# # # for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
# # #     buildinginfo[col] = pd.to_numeric(buildinginfo[col].replace('-', np.nan))

# # # # train/test에 building_info 병합
# # # train = train.merge(buildinginfo, on='건물번호', how='left')
# # # test = test.merge(buildinginfo, on='건물번호', how='left')

# # # # 범주형 -> 숫자형 변환
# # # for df in [train, test]:
# # #     df['건물유형'] = df['건물유형'].astype('category').cat.codes

# # # # 공통 feature 정의 (test 기준)
# # # test_features = ['건물번호', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)',
# # #                  'ESS저장용량(kWh)', 'PCS용량(kW)', '기온(°C)', '강수량(mm)', '풍속(m/s)',
# # #                  '습도(%)', 'hour', 'dayofweek']

# # # # train feature는 '일조(hr)', '일사(MJ/m2)' 포함되므로 따로 정의
# # # train_features = test_features + ['일조(hr)', '일사(MJ/m2)']
# # # target = '전력소비량(kWh)'

# # # x = train[train_features]
# # # y = train[target]
# # # x_test_final = test[test_features]

# # # # 결측치 채우기
# # # x = x.fillna(0)
# # # x_test_final = x_test_final.fillna(0)

# # # # 학습/검증 분리
# # # x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=seed)

# # # # 스케일링 (test feature에 맞춰서만 적용)
# # # scaler = StandardScaler()
# # # x_train_scaled = scaler.fit_transform(x_train[test_features])
# # # x_val_scaled = scaler.transform(x_val[test_features])
# # # x_test_final_scaled = scaler.transform(x_test_final)

# # # # 부스팅 모델 정의
# # # xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=seed)
# # # lgb_model = LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=seed)
# # # cat_model = CatBoostRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=seed, verbose=0)

# # # # 스태킹 앙상블 정의
# # # stack_model = StackingRegressor(
# # #     estimators=[
# # #         ('xgb', xgb_model),
# # #         ('lgb', lgb_model),
# # #         ('cat', cat_model),
# # #     ],
# # #     final_estimator=GradientBoostingRegressor(n_estimators=100, random_state=seed),
# # #     n_jobs=-1
# # # )

# # # # 모델 학습
# # # stack_model.fit(x_train_scaled, y_train)

# # # # 검증 예측 및 SMAPE
# # # val_pred = stack_model.predict(x_val_scaled)
# # # def smape(y_true, y_pred):
# # #     denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
# # #     diff = np.abs(y_true - y_pred) / denominator
# # #     diff[denominator == 0] = 0.0
# # #     return np.mean(diff) * 100

# # # print(f"✅ 검증 SMAPE: {smape(y_val, val_pred):.4f}")

# # # # 테스트 예측 후 제출
# # # final_pred = stack_model.predict(x_test_final_scaled)
# # # samplesubmission['answer'] = final_pred
# # # filename = f"submission_stack_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
# # # samplesubmission.to_csv(os.path.join(path, filename), index=False)
# # # print(f"📁 저장 완료 → {filename}")


# # # import os
# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.preprocessing import StandardScaler
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
# # # from sklearn.metrics import mean_absolute_error
# # # from xgboost import XGBRegressor
# # # from catboost import CatBoostRegressor
# # # from lightgbm import LGBMRegressor
# # # import random
# # # import tensorflow as tf
# # # import datetime

# # # # Seed 고정
# # # seed = 42
# # # random.seed(seed)
# # # np.random.seed(seed)
# # # tf.random.set_seed(seed)

# # # # 경로 설정
# # # if os.path.exists('/workspace/TensorJae/Study25/'):
# # #     BASE_PATH = '/workspace/TensorJae/Study25/'
# # # else:
# # #     BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')

# # # path = os.path.join(BASE_PATH, '_data/dacon/electricity/')
# # # save_path = os.path.join(path, '_save/')

# # # # 데이터 불러오기
# # # buildinginfo = pd.read_csv(path + 'building_info.csv')
# # # train = pd.read_csv(path + 'train.csv')
# # # test = pd.read_csv(path + 'test.csv')
# # # samplesub = pd.read_csv(path + 'sample_submission.csv')

# # # # 결측치 처리: '-' → 0, float 변환
# # # cols_to_clean = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
# # # for col in cols_to_clean:
# # #     buildinginfo[col] = buildinginfo[col].replace('-', 0).astype(float)

# # # # 날짜 파싱 및 파생변수
# # # for df in [train, test]:
# # #     df['일시'] = pd.to_datetime(df['일시'])
# # #     df['hour'] = df['일시'].dt.hour
# # #     df['dayofweek'] = df['일시'].dt.dayofweek

# # # # 건물 정보 병합
# # # train = train.merge(buildinginfo, on='건물번호', how='left')
# # # test = test.merge(buildinginfo, on='건물번호', how='left')

# # # # 범주형 처리
# # # train['건물유형'] = train['건물유형'].astype('category').cat.codes
# # # test['건물유형'] = test['건물유형'].astype('category').cat.codes

# # # # 피처 설정
# # # test_features = ['건물번호', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)',
# # #                  'ESS저장용량(kWh)', 'PCS용량(kW)', '기온(°C)', '강수량(mm)', '풍속(m/s)',
# # #                  '습도(%)', 'hour', 'dayofweek']
# # # target = '전력소비량(kWh)'

# # # # 학습/테스트 분할
# # # x = train[test_features]
# # # y = np.log1p(train[target])
# # # x_test_final = test[test_features]

# # # x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=seed)

# # # # 스케일링
# # # scaler = StandardScaler()
# # # x_train_scaled = scaler.fit_transform(x_train)
# # # x_val_scaled = scaler.transform(x_val)
# # # x_test_final_scaled = scaler.transform(x_test_final)

# # # # 모델 정의
# # # xgb = XGBRegressor(n_estimators=600, learning_rate=0.1, max_depth=6, random_state=seed)
# # # lgb = LGBMRegressor(n_estimators=600, learning_rate=0.1, max_depth=6, random_state=seed)
# # # cat = CatBoostRegressor(n_estimators=600, learning_rate=0.1, max_depth=6, random_state=seed, verbose=0)

# # # stack_model = StackingRegressor(
# # #     estimators=[
# # #         ('xgb', xgb),
# # #         ('lgb', lgb),
# # #         ('cat', cat),
# # #     ],
# # #     final_estimator=GradientBoostingRegressor(n_estimators=350, random_state=seed),
# # #     n_jobs=-1
# # # )

# # # # 학습
# # # stack_model.fit(x_train_scaled, y_train)

# # # # 예측 (로그 복원)
# # # y_pred = np.expm1(stack_model.predict(x_val_scaled))
# # # y_true = np.expm1(y_val)

# # # # SMAPE 계산
# # # def smape(y_true, y_pred):
# # #     return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

# # # print(f"\n✅ 검증 SMAPE: {smape(y_true, y_pred):.4f}")

# # # # 제출
# # # # 예측 및 로그 복원
# # # final_pred = np.expm1(stack_model.predict(x_test_final_scaled))
# # # samplesub['answer'] = final_pred

# # # # 오늘 날짜
# # # today = datetime.datetime.now().strftime('%Y%m%d')

# # # # 검증 SMAPE 점수 계산
# # # val_smape = smape(y_true, y_pred)
# # # score_str = f"{val_smape:.4f}".replace('.', '_')

# # # # 파일명 생성
# # # filename = f"submission_{today}_SMAPE_{score_str}.csv"
# # # file_path = os.path.join(path, filename)

# # # # 저장
# # # samplesub.to_csv(file_path, index=False)
# # # print(f"📁 {filename} 저장 완료!")

























# # import os
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.ensemble import GradientBoostingRegressor
# # from sklearn.linear_model import RidgeCV
# # from sklearn.metrics import mean_absolute_error
# # from xgboost import XGBRegressor
# # from lightgbm import LGBMRegressor
# # from catboost import CatBoostRegressor
# # import random
# # import tensorflow as tf
# # import datetime

# # # Seed 고정
# # seed = 42
# # random.seed(seed)
# # np.random.seed(seed)
# # tf.random.set_seed(seed)

# # # 경로 설정
# # if os.path.exists('/workspace/TensorJae/Study25/'):
# #     BASE_PATH = '/workspace/TensorJae/Study25/'
# # else:
# #     BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')

# # path = os.path.join(BASE_PATH, '_data/dacon/electricity/')
# # save_path = os.path.join(path, '_save/')

# # # 데이터 로드
# # buildinginfo = pd.read_csv(path + 'building_info.csv')
# # train = pd.read_csv(path + 'train.csv')
# # test = pd.read_csv(path + 'test.csv')
# # samplesub = pd.read_csv(path + 'sample_submission.csv')

# # # '-' 결측치 처리 및 float 변환
# # for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
# #     buildinginfo[col] = buildinginfo[col].replace('-', 0).astype(float)

# # # 날짜 파싱 및 파생변수
# # for df in [train, test]:
# #     df['일시'] = pd.to_datetime(df['일시'])
# #     df['hour'] = df['일시'].dt.hour
# #     df['dayofweek'] = df['일시'].dt.dayofweek

# # # 병합
# # train = train.merge(buildinginfo, on='건물번호', how='left')
# # test = test.merge(buildinginfo, on='건물번호', how='left')

# # # 범주형 → 수치형
# # train['건물유형'] = train['건물유형'].astype('category').cat.codes
# # test['건물유형'] = test['건물유형'].astype('category').cat.codes

# # # Feature 정의
# # test_features = ['건물번호', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)',
# #                  'ESS저장용량(kWh)', 'PCS용량(kW)', '기온(°C)', '강수량(mm)', '풍속(m/s)',
# #                  '습도(%)', 'hour', 'dayofweek']
# # target = '전력소비량(kWh)'

# # # 데이터 분할 및 전처리
# # x = train[test_features]
# # y = np.log1p(train[target])
# # x_test_final = test[test_features]

# # x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=seed)

# # scaler = StandardScaler()
# # x_train_scaled = scaler.fit_transform(x_train)
# # x_val_scaled = scaler.transform(x_val)
# # x_test_final_scaled = scaler.transform(x_test_final)

# # # ✅ 베이스 모델 정의 + 조기 종료
# # xgb_model = XGBRegressor(
# #     n_estimators=1000,
# #     learning_rate=0.05,
# #     max_depth=6,
# #     random_state=seed,
# #     early_stopping_rounds=50,
# #     objective='reg:squarederror'
# # )
# # xgb_model.fit(
# #     x_train_scaled, y_train,
# #     eval_set=[(x_val_scaled, y_val)],
# #     verbose=False
# # )

# # lgb_model = LGBMRegressor(
# #     n_estimators=1000,
# #     learning_rate=0.05,
# #     max_depth=6,
# #     random_state=seed,
# #     objective='mae'
# # )
# # lgb_model.fit(
# #     x_train_scaled, y_train,
# #     eval_set=[(x_val_scaled, y_val)],
# #     early_stopping_rounds=50,
# #     verbose=-1
# # )

# # cat_model = CatBoostRegressor(
# #     n_estimators=1000,
# #     learning_rate=0.05,
# #     max_depth=6,
# #     random_seed=seed,
# #     verbose=0,
# #     loss_function='MAE'
# # )
# # cat_model.fit(
# #     x_train_scaled, y_train,
# #     eval_set=(x_val_scaled, y_val),
# #     early_stopping_rounds=50
# # )

# # # ✅ 스태킹용 예측 결과
# # oof_train = np.vstack([
# #     xgb_model.predict(x_train_scaled),
# #     lgb_model.predict(x_train_scaled),
# #     cat_model.predict(x_train_scaled)
# # ]).T

# # oof_val = np.vstack([
# #     xgb_model.predict(x_val_scaled),
# #     lgb_model.predict(x_val_scaled),
# #     cat_model.predict(x_val_scaled)
# # ]).T

# # oof_test = np.vstack([
# #     xgb_model.predict(x_test_final_scaled),
# #     lgb_model.predict(x_test_final_scaled),
# #     cat_model.predict(x_test_final_scaled)
# # ]).T

# # # ✅ 메타 모델 학습
# # from sklearn.ensemble import GradientBoostingRegressor

# # meta_model = GradientBoostingRegressor(
# #     n_estimators=700, learning_rate=0.05, max_depth=3, random_state=seed
# # )
# # meta_model.fit(oof_train, y_train)

# # # 검증 예측
# # val_pred = meta_model.predict(oof_val)
# # y_val_exp = np.expm1(y_val)
# # val_pred_exp = np.expm1(val_pred)

# # # SMAPE 정의
# # def smape(y_true, y_pred):
# #     return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

# # val_smape = smape(y_val_exp, val_pred_exp)
# # print(f"\n✅ 스태킹 SMAPE: {val_smape:.4f}")

# # # 최종 예측
# # final_pred_log = meta_model.predict(oof_test)
# # final_pred = np.expm1(final_pred_log)

# # # 저장
# # samplesub['answer'] = final_pred
# # today = datetime.datetime.now().strftime('%Y%m%d')
# # score_str = f"{val_smape:.4f}".replace('.', '_')
# # filename = f"submission_{today}_SMAPE_{score_str}.csv"
# # samplesub.to_csv(os.path.join(path, filename), index=False)
# # print(f"📁 저장 완료 → {filename}")



# # # GradientRegressor nestimator 700> 1000 























# # import os
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from xgboost import XGBRegressor
# # from lightgbm import LGBMRegressor
# # from catboost import CatBoostRegressor
# # from sklearn.ensemble import GradientBoostingRegressor
# # import lightgbm as lgb
# # import random
# # import tensorflow as tf
# # import datetime

# # # Seed 고정

# # # seed best so far !!!!!! 42
# # seed = 707
# # random.seed(seed)
# # np.random.seed(seed)
# # tf.random.set_seed(seed)

# # # 경로 설정
# # if os.path.exists('/workspace/TensorJae/Study25/'):
# #     BASE_PATH = '/workspace/TensorJae/Study25/'
# # else:
# #     BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')

# # path = os.path.join(BASE_PATH, '_data/dacon/electricity/')
# # save_path = os.path.join(path, '_save/')

# # # 데이터 로드
# # buildinginfo = pd.read_csv(path + 'building_info.csv')
# # train = pd.read_csv(path + 'train.csv')
# # test = pd.read_csv(path + 'test.csv')
# # samplesub = pd.read_csv(path + 'sample_submission.csv')

# # # 결측치 처리
# # for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
# #     buildinginfo[col] = buildinginfo[col].replace('-', 0).astype(float)

# # def feature_engineering(df):
# #     df = df.copy()
# #     df['일시'] = pd.to_datetime(df['일시'])
# #     df['hour'] = df['일시'].dt.hour
# #     df['dayofweek'] = df['일시'].dt.dayofweek
# #     df['month'] = df['일시'].dt.month
# #     df['day'] = df['일시'].dt.day
# #     df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
# #     df['is_working_hours'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 18 else 0)
# #     df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
# #     df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
# #     for col in ['일조(hr)', '일사(MJ/m2)']:
# #         if col in df.columns:
# #             df[col] = df[col].fillna(0)
# #     temp = df['기온(°C)']
# #     humidity = df['습도(%)']
# #     df['DI'] = 9/5 * temp - 0.55 * (1 - humidity/100) * (9/5 * temp - 26) + 32
# #     return df

# # print("📊 Feature Engineering...")
# # train = feature_engineering(train)
# # test = feature_engineering(test)

# # train = train.merge(buildinginfo, on='건물번호', how='left')
# # test = test.merge(buildinginfo, on='건물번호', how='left')
# # train['건물유형'] = train['건물유형'].astype('category').cat.codes
# # test['건물유형'] = test['건물유형'].astype('category').cat.codes

# # features = [
# #     '건물번호', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)',
# #     'ESS저장용량(kWh)', 'PCS용량(kW)', '기온(°C)', '강수량(mm)', '풍속(m/s)',
# #     '습도(%)', 'hour', 'dayofweek', 'month', 'day',
# #     'is_weekend', 'is_working_hours', 'sin_hour', 'cos_hour', 'DI'
# # ]
# # target = '전력소비량(kWh)'

# # x = train[features] 
# # y = np.log1p(train[target])
# # x_test_final = test[features]

# # x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=seed)

# # scaler = StandardScaler()
# # x_train_scaled = scaler.fit_transform(x_train)
# # x_val_scaled = scaler.transform(x_val)
# # x_test_final_scaled = scaler.transform(x_test_final)

# # # 모델 학습
# # print("🚀 Training base models...")
# # xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=seed,
# #                          early_stopping_rounds=50, objective='reg:squarederror')
# # xgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

# # lgb_model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=seed, objective='mae')
# # lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

# # cat_model = CatBoostRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6,
# #                               random_seed=seed, verbose=0, loss_function='MAE')
# # cat_model.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50)

# # # OOF 생성
# # oof_train = np.vstack([
# #     xgb_model.predict(x_train_scaled),
# #     lgb_model.predict(x_train_scaled),
# #     cat_model.predict(x_train_scaled)
# # ]).T

# # oof_val = np.vstack([
# #     xgb_model.predict(x_val_scaled),
# #     lgb_model.predict(x_val_scaled),
# #     cat_model.predict(x_val_scaled)
# # ]).T

# # oof_test = np.vstack([
# #     xgb_model.predict(x_test_final_scaled),
# #     lgb_model.predict(x_test_final_scaled),
# #     cat_model.predict(x_test_final_scaled)
# # ]).T

# # # 메타 모델
# # print("🔁 Meta model training...")
# # meta_model = GradientBoostingRegressor(n_estimators=700, learning_rate=0.05, max_depth=3, random_state=seed)
# # meta_model.fit(oof_train, y_train)

# # val_pred = meta_model.predict(oof_val)
# # y_val_exp = np.expm1(y_val)
# # val_pred_exp = np.expm1(val_pred)

# # # ✅ 정확한 SMAPE 구현
# # def smape(y_true, y_pred):
# #     epsilon = 1e-6  # 0으로 나누기 방지
# #     denominator = (np.abs(y_true) + np.abs(y_pred)) + epsilon
# #     return 100 * np.mean(2.0 * np.abs(y_pred - y_true) / denominator)

# # val_smape = smape(y_val_exp, val_pred_exp)
# # print(f"\n✅ Stacking SMAPE: {val_smape:.4f}")

# # # 최종 예측
# # final_pred_log = meta_model.predict(oof_test)
# # final_pred = np.expm1(final_pred_log)

# # samplesub['answer'] = final_pred
# # today = datetime.datetime.now().strftime('%Y%m%d')
# # score_str = f"{val_smape:.4f}".replace('.', '_')
# # filename = f"submission_{today}_SMAPE_{score_str}.csv"
# # samplesub.to_csv(os.path.join(path, filename), index=False)
# # print(f"📁 저장 완료 → {filename}")

# import os
# import pandas as pd
# import numpy as np
# import random
# import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.linear_model import RidgeCV
# from sklearn.metrics import mean_absolute_error

# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
# import lightgbm as lgb
# import tensorflow as tf

# # Seed 고정
# seed = 43


# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)

# # 경로 설정
# if os.path.exists('/workspace/TensorJae/Study25/'):
#     BASE_PATH = '/workspace/TensorJae/Study25/'
# else:
#     BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')

# path = os.path.join(BASE_PATH, '_data/dacon/electricity/')

# # 데이터 로드
# buildinginfo = pd.read_csv(path + 'building_info.csv')
# train = pd.read_csv(path + 'train.csv')
# test = pd.read_csv(path + 'test.csv')
# samplesub = pd.read_csv(path + 'sample_submission.csv')

# # 결측치 처리
# for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
#     buildinginfo[col] = buildinginfo[col].replace('-', 0).astype(float)

# # Feature Engineering
# def feature_engineering(df):
#     df = df.copy()
#     df['일시'] = pd.to_datetime(df['일시'])
#     df['hour'] = df['일시'].dt.hour
#     df['dayofweek'] = df['일시'].dt.dayofweek
#     df['month'] = df['일시'].dt.month
#     df['day'] = df['일시'].dt.day
#     df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
#     df['is_working_hours'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 18 else 0)
#     df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
#     df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
#     for col in ['일조(hr)', '일사(MJ/m2)']:
#         if col in df.columns:
#             df[col] = df[col].fillna(0)
#     temp = df['기온(°C)']
#     humidity = df['습도(%)']
#     df['DI'] = 9/5 * temp - 0.55 * (1 - humidity/100) * (9/5 * temp - 26) + 32
#     return df

# # 전처리
# train = feature_engineering(train)
# test = feature_engineering(test)
# train = train.merge(buildinginfo, on='건물번호', how='left')
# test = test.merge(buildinginfo, on='건물번호', how='left')
# train['건물유형'] = train['건물유형'].astype('category').cat.codes
# test['건물유형'] = test['건물유형'].astype('category').cat.codes

# features = [
#     '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
#     '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
#     'hour', 'dayofweek', 'month', 'day', 'is_weekend',
#     'is_working_hours', 'sin_hour', 'cos_hour', 'DI'
# ]

# target = '전력소비량(kWh)'

# # 최종 예측 결과 저장용
# final_preds = []
# val_smapes = []

# # 건물별로 모델 학습 및 예측
# building_ids = train['건물번호'].unique()

# for bno in building_ids:
#     print(f"🏢 건물번호 {bno} 모델링 중...")

#     train_b = train[train['건물번호'] == bno].copy()
#     test_b = test[test['건물번호'] == bno].copy()

#     x = train_b[features]
#     y = np.log1p(train_b[target])
#     x_test_final = test_b[features]

#     x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=seed)

#     scaler = StandardScaler()
#     x_train_scaled = scaler.fit_transform(x_train)
#     x_val_scaled = scaler.transform(x_val)
#     x_test_final_scaled = scaler.transform(x_test_final)

#     # Base models
#     xgb_model = XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
#                              random_state=seed, early_stopping_rounds=50, objective='reg:squarederror')
#     xgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

#     lgb_model = LGBMRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
#                               random_state=seed, objective='mae')
#     lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],
#                   callbacks=[lgb.early_stopping(50, verbose=False)])

#     cat_model = CatBoostRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
#                                   random_seed=seed, verbose=0, loss_function='MAE')
#     cat_model.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50)

#     # Level 1 predictions
#     oof_train_lvl1 = np.vstack([
#         xgb_model.predict(x_train_scaled),
#         lgb_model.predict(x_train_scaled),
#         cat_model.predict(x_train_scaled)
#     ]).T
#     oof_val_lvl1 = np.vstack([
#         xgb_model.predict(x_val_scaled),
#         lgb_model.predict(x_val_scaled),
#         cat_model.predict(x_val_scaled)
#     ]).T
#     oof_test_lvl1 = np.vstack([
#         xgb_model.predict(x_test_final_scaled),
#         lgb_model.predict(x_test_final_scaled),
#         cat_model.predict(x_test_final_scaled)
#     ]).T

#     # Level 2 Meta model
#     meta_model = RidgeCV()
#     meta_model.fit(oof_train_lvl1, y_train)
#     val_pred_lvl2 = meta_model.predict(oof_val_lvl1)
#     test_pred_lvl2 = meta_model.predict(oof_test_lvl1)

#     # Level 3 Final model
#     final_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=seed)
#     final_model.fit(val_pred_lvl2.reshape(-1, 1), y_val)

#     val_final = final_model.predict(val_pred_lvl2.reshape(-1, 1))
#     val_smape = np.mean(200 * np.abs(np.expm1(val_final) - np.expm1(y_val)) /
#                         (np.abs(np.expm1(val_final)) + np.abs(np.expm1(y_val)) + 1e-6))
#     val_smapes.append(val_smape)

#     pred = np.expm1(final_model.predict(test_pred_lvl2.reshape(-1, 1)))
#     final_preds.extend(pred)

# # 결과 저장
# samplesub['answer'] = final_preds
# today = datetime.datetime.now().strftime('%Y%m%d')
# avg_smape = np.mean(val_smapes)
# score_str = f"{avg_smape:.4f}".replace('.', '_')
# filename = f"submission_groupwise_{today}_SMAPE_{score_str}_{seed}.csv"
# samplesub.to_csv(os.path.join(path, filename), index=False)
# print(f"\n📁 저장 완료 → {filename}")
# print(f"✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")


# # import os
# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
# # from xgboost import XGBRegressor
# # from catboost import CatBoostRegressor
# # from lightgbm import LGBMRegressor
# # import random
# # import tensorflow as tf
# # import datetime

# # # Seed 고정
# # seed = 42
# # random.seed(seed)
# # np.random.seed(seed)
# # tf.random.set_seed(seed)

# # # 경로 설정
# # BASE_PATH = './' # 사용자의 환경에 맞게 조절하세요.
# # path = os.path.join(BASE_PATH, '_data/dacon/electricity/')

# # # 데이터 불러오기
# # buildinginfo = pd.read_csv(path + 'building_info.csv', encoding='cp949')
# # train = pd.read_csv(path + 'train.csv', encoding='cp949')
# # test = pd.read_csv(path + 'test.csv', encoding='cp949')
# # samplesub = pd.read_csv(path + 'sample_submission.csv', encoding='cp949')

# # # --- 1. 기본 전처리 ---
# # # 결측치 처리 및 타입 변환
# # for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
# #     buildinginfo[col] = buildinginfo[col].replace('-', 0).astype(float)

# # # 날짜 파싱 및 기본 시간 특성 생성
# # for df in [train, test]:
# #     df['일시'] = pd.to_datetime(df['일시'])

# # # 건물 정보 병합
# # train = train.merge(buildinginfo, on='건물번호', how='left')
# # test = test.merge(buildinginfo, on='건물번호', how='left')

# # # 범주형 처리
# # train['건물유형'] = train['건물유형'].astype('category').cat.codes
# # test['건물유형'] = test['건물유형'].astype('category').cat.codes

# # # --- 2. [고도화] 시계열 특성 생성 (Lag & Rolling) ---
# # # 훈련 데이터와 테스트 데이터를 합쳐서 특성 생성 (데이터 일관성 유지)
# # combined_df = pd.concat([train, test], ignore_index=True)
# # combined_df = combined_df.sort_values(by=['건물번호', '일시']).reset_index(drop=True)

# # # 로그 변환된 타겟 생성 (이후 이동평균/시차 특성에 사용)
# # combined_df['log_target'] = np.log1p(combined_df['전력소비량(kWh)'])

# # # 시차(Lag) 특성 생성
# # lags = [24, 48, 168] # 1일, 2일, 1주일 전
# # for lag in lags:
# #     combined_df[f'lag_{lag}'] = combined_df.groupby('건물번호')['log_target'].shift(lag)

# # # 이동 평균(Rolling) 특성 생성
# # windows = [24, 48, 168]
# # for window in windows:
# #     combined_df[f'rolling_mean_{window}'] = combined_df.groupby('건물번호')['log_target'].transform(
# #         lambda x: x.shift(24).rolling(window=window, min_periods=1).mean() # shift(24)로 24시간 전부터의 평균 계산
# #     )
# #     combined_df[f'rolling_max_{window}'] = combined_df.groupby('건물번호')['log_target'].transform(
# #         lambda x: x.shift(24).rolling(window=window, min_periods=1).max()
# #     )

# # # Featuretools에서 영감을 받은 시간 관련 특성 추가
# # combined_df['hour'] = combined_df['일시'].dt.hour
# # combined_df['dayofweek'] = combined_df['일시'].dt.dayofweek
# # combined_df['month'] = combined_df['일시'].dt.month
# # combined_df['dayofyear'] = combined_df['일시'].dt.dayofyear

# # # 특성 생성 후 다시 훈련/테스트 데이터로 분리
# # train = combined_df[combined_df['전력소비량(kWh)'].notna()].copy()
# # test = combined_df[combined_df['전력소비량(kWh)'].isna()].copy()

# # # 생성된 특성에서 발생한 NaN 값 처리 (과거 데이터가 없는 경우)
# # # bfill: 뒤의 값으로 채움. 훈련 초반 데이터에만 해당.
# # train = train.fillna(method='bfill')

# # # --- 3. 피처 및 타겟 설정 ---
# # features = [
# #     '건물번호', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)',
# #     'ESS저장용량(kWh)', 'PCS용량(kW)', '기온(°C)', '강수량(mm)', '풍속(m/s)',
# #     '습도(%)', 'hour', 'dayofweek', 'month', 'dayofyear',
# # ]
# # # 생성한 시계열 특성 추가
# # features.extend([col for col in train.columns if 'lag_' in col or 'rolling_' in col])
# # target = '전력소비량(kWh)'

# # x = train[features]
# # y = np.log1p(train[target])
# # x_test_final = test[features]

# # # --- 4. [고도화] 시계열 교차 검증 분할 ---
# # split_date = pd.to_datetime('2022-08-18') # 훈련 기간의 마지막 주 시작일
# # val_indices = train[train['일시'] >= split_date].index

# # x_train, x_val = x.drop(val_indices), x.loc[val_indices]
# # y_train, y_val = y.drop(val_indices), y.loc[val_indices]

# # # --- 5. 스케일링 및 모델 학습 ---
# # scaler = StandardScaler()
# # x_train_scaled = scaler.fit_transform(x_train)
# # x_val_scaled = scaler.transform(x_val)
# # x_test_final_scaled = scaler.transform(x_test_final)

# # # 모델 정의
# # xgb = XGBRegressor(n_estimators=600, learning_rate=0.1, max_depth=6, random_state=seed, n_jobs=-1)
# # lgb = LGBMRegressor(n_estimators=600, learning_rate=0.1, max_depth=6, random_state=seed, n_jobs=-1)
# # cat = CatBoostRegressor(n_estimators=600, learning_rate=0.1, max_depth=6, random_state=seed, verbose=0)

# # stack_model = StackingRegressor(
# #     estimators=[('xgb', xgb), ('lgb', lgb), ('cat', cat)],
# #     final_estimator=GradientBoostingRegressor(n_estimators=350, random_state=seed),
# #     n_jobs=-1,
# #     cv='passthrough' # 훈련 데이터를 final_estimator에 그대로 사용
# # )

# # print("Starting model training...")
# # stack_model.fit(x_train_scaled, y_train)

# # # --- 6. 평가 및 제출 ---
# # def smape(y_true, y_pred):
# #     return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

# # y_pred_log = stack_model.predict(x_val_scaled)
# # y_pred_exp = np.expm1(y_pred_log)
# # y_true_exp = np.expm1(y_val)

# # val_smape = smape(y_true_exp, y_pred_exp)
# # print(f"\n✅ Validation SMAPE: {val_smape:.4f}")

# # final_pred_log = stack_model.predict(x_test_final_scaled)
# # final_pred_exp = np.expm1(final_pred_log)

# # samplesub['answer'] = final_pred_exp

# # # 파일 저장
# # today = datetime.datetime.now().strftime('%Y%m%d')
# # score_str = f"{val_smape:.4f}".replace('.', '_')
# # filename = f"submission_{today}_SMAPE_{score_str}.csv"
# # file_path = os.path.join('./', filename) # 저장 경로
# # samplesub.to_csv(file_path, index=False)
# # print(f"📁 {filename} saved successfully!")


import os
import pandas as pd
import numpy as np
import random
import datetime
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import tensorflow as tf
import warnings
warnings.filterwarnings(action='ignore')

# Seed 고정
seed = 707
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 경로 설정
if os.path.exists('/workspace/TensorJae/Study25/'):
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')

path = os.path.join(BASE_PATH, '_data/dacon/electricity/')
buildinginfo = pd.read_csv(path + 'building_info.csv')
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
samplesub = pd.read_csv(path + 'sample_submission.csv')

# 결측치 처리
for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    buildinginfo[col] = buildinginfo[col].replace('-', 0).astype(float)

# Feature Engineering
def feature_engineering(df):
    df = df.copy()
    df['일시'] = pd.to_datetime(df['일시'])
    df['hour'] = df['일시'].dt.hour
    df['dayofweek'] = df['일시'].dt.dayofweek
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_working_hours'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 18 else 0)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    for col in ['일조(hr)', '일사(MJ/m2)']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    temp = df['기온(°C)']
    humidity = df['습도(%)']
    df['DI'] = 9/5 * temp - 0.55 * (1 - humidity/100) * (9/5 * temp - 26) + 32
    return df

train = feature_engineering(train)
test = feature_engineering(test)
train = train.merge(buildinginfo, on='건물번호', how='left')
test = test.merge(buildinginfo, on='건물번호', how='left')
train['건물유형'] = train['건물유형'].astype('category').cat.codes
test['건물유형'] = test['건물유형'].astype('category').cat.codes

all_features = [
    '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
    'hour', 'dayofweek', 'month', 'day', 'is_weekend',
    'is_working_hours', 'sin_hour', 'cos_hour', 'DI'
]

target = '전력소비량(kWh)'
final_preds = []
val_smapes = []

# 중요도 기반 feature filtering (기본 xgb 기준)
xgb_temp = XGBRegressor()
x = train[all_features]
y = np.log1p(train[target])
xgb_temp.fit(x, y)
importances = xgb_temp.feature_importances_
feature_importance_dict = dict(zip(all_features, importances))
selected_features = [f for f, score in feature_importance_dict.items() if score >= 0.01]
print(f"\n✅ 선택된 feature ({len(selected_features)}개): {selected_features}")

# Optuna 튜닝 함수
def objective(trial, oof_train, oof_val, y_train, y_val):
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    ridge = Ridge(alpha=alpha)
    ridge.fit(oof_train, y_train)
    preds = ridge.predict(oof_val)
    smape = np.mean(200 * np.abs(np.expm1(preds) - np.expm1(y_val)) /
                    (np.abs(np.expm1(preds)) + np.abs(np.expm1(y_val)) + 1e-6))
    return smape

# 건물별 모델 학습
building_ids = train['건물번호'].unique()
for bno in building_ids:
    print(f"\n🏢 건물번호 {bno} 처리 중...")

    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()
    x = train_b[selected_features]
    y = np.log1p(train_b[target])
    x_test_final = test_b[selected_features]

    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=seed)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_final_scaled = scaler.transform(x_test_final)

    # Base 모델 학습
    xgb = XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                       random_state=seed, early_stopping_rounds=50, objective='reg:squarederror')
    xgb.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

    lgb_model = LGBMRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                              random_state=seed, objective='mae')
    lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

    cat = CatBoostRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                            random_seed=seed, verbose=0, loss_function='MAE')
    cat.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50)

    # 스태킹
    oof_train_lvl1 = np.vstack([
        xgb.predict(x_train_scaled),
        lgb_model.predict(x_train_scaled),
        cat.predict(x_train_scaled)
    ]).T
    oof_val_lvl1 = np.vstack([
        xgb.predict(x_val_scaled),
        lgb_model.predict(x_val_scaled),
        cat.predict(x_val_scaled)
    ]).T
    oof_test_lvl1 = np.vstack([
        xgb.predict(x_test_final_scaled),
        lgb_model.predict(x_test_final_scaled),
        cat.predict(x_test_final_scaled)
    ]).T

    # Optuna로 Ridge 튜닝
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, oof_train_lvl1, oof_val_lvl1, y_train, y_val), n_trials=30)
    best_alpha = study.best_params['alpha']
    meta_model = Ridge(alpha=best_alpha)
    meta_model.fit(oof_train_lvl1, y_train)

    val_pred = meta_model.predict(oof_val_lvl1)
    test_pred = meta_model.predict(oof_test_lvl1)

    val_smape = np.mean(200 * np.abs(np.expm1(val_pred) - np.expm1(y_val)) /
                        (np.abs(np.expm1(val_pred)) + np.abs(np.expm1(y_val)) + 1e-6))
    val_smapes.append(val_smape)

    pred = np.expm1(test_pred)
    final_preds.extend(pred)

# 결과 저장
samplesub['answer'] = final_preds
today = datetime.datetime.now().strftime('%Y%m%d')
avg_smape = np.mean(val_smapes)
score_str = f"{avg_smape:.4f}".replace('.', '_')
filename = f"submission_stack_filtered_{today}_SMAPE_{score_str}_{seed}.csv"
samplesub.to_csv(os.path.join(path, filename), index=False)

print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
print(f"📁 저장 완료 → {filename}")