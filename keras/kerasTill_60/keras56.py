# # # from matplotlib.pylab import RandomState
# # # import numpy as np
# # # from keras.models import Sequential
# # # from keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout, BatchNormalization
# # # import pandas as pd
# # # from sklearn.metrics import mean_squared_error

# # # import time

# # # from sklearn.model_selection import train_test_split

# # # # 2016/12/31 00:10 ~ 2017/01/01 00:00
# # # file_path = '/workspace/TensorJae/Study25/_data/jena_climate_2009_2016.csv'
# # # # csv_file = 'jena_climate_2009_2016.csv'

# # # # 날짜 컬럼 기준으로 필요한 행만 추출하면서 불러오기
# # # use_cols = ['Date Time', 'wd (deg)']  # 일단 컬럼명 확인용

# # # # 1. 전체 헤더만 미리 확인
# # # df_header = pd.read_csv(file_path, nrows=5)
# # # print(df_header.columns)

# # # # Index(['Date Time', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)',
# # # #        'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
# # # #        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
# # # #        'wd (deg)'],
# # # #       dtype='object')

# # # df = pd.read_csv(file_path, parse_dates=['Date Time'])

# # # # start = pd.to_datetime()

# # # # 예측할 구간
# # # start_pred = pd.to_datetime("31.12.2016 00:10:00", dayfirst=True)
# # # end_pred = pd.to_datetime("01.01.2017 00:00:00", dayfirst=True)

# # # # ✅ 학습에 쓸 데이터는 이 구간을 제외한 나머지
# # # df_train = df[~((df['Date Time'] >= start_pred) & (df['Date Time'] <= end_pred))].copy()

# # # # print(df_train.head())
# # # # print(df_train.tail())

# # # #   Date Time  p (mbar)  T (degC)  Tpot (K)  ...  rho (g/m**3)  wv (m/s)  max. wv (m/s)  wd (deg)
# # # # 0 2009-01-01 00:10:00    996.52     -8.02    265.40  ...       1307.75      1.03           1.75     152.3
# # # # 1 2009-01-01 00:20:00    996.57     -8.41    265.01  ...       1309.80      0.72           1.50     136.1
# # # # 2 2009-01-01 00:30:00    996.53     -8.51    264.91  ...       1310.24      0.19           0.63     171.6
# # # # 3 2009-01-01 00:40:00    996.51     -8.31    265.12  ...       1309.19      0.34           0.50     198.0
# # # # 4 2009-01-01 00:50:00    996.51     -8.27    265.15  ...       1309.00      0.32           0.63     214.3

# # # # [5 rows x 15 columns]
# # # #                  Date Time  p (mbar)  T (degC)  Tpot (K)  ...  rho (g/m**3)  wv (m/s)  max. wv (m/s)  wd (deg)
# # # # 420402 2016-12-30 23:20:00   1008.80     -3.59    268.89  ...       1301.77      1.11           1.56     192.7
# # # # 420403 2016-12-30 23:30:00   1008.79     -3.71    268.77  ...       1302.25      0.77           1.40     146.8
# # # # 420404 2016-12-30 23:40:00   1008.68     -3.89    268.60  ...       1302.94      0.97           1.54     170.6
# # # # 420405 2016-12-30 23:50:00   1008.67     -4.02    268.47  ...       1303.56      0.72           1.74     186.7
# # # # 420406 2016-12-31 00:00:00   1008.67     -4.09    268.41  ...       1303.88      0.97           1.78     169.5

# # # # [5 rows x 15 columns]

# # # timesteps = 24

# # # # x에 사용할 feature들 (wd (deg) 제외)
# # # feature_columns = [col for col in df_train.columns if col not in ['Date Time', 'wd (deg)']]
# # # target_column = 'wd (deg)'

# # # # numpy 배열로 변환
# # # x_data = df_train[feature_columns].values
# # # y_data = df_train[target_column].values

# # # # 슬라이딩 윈도우 함수
# # # def split_xy(x, y, timesteps):
# # #     x_list, y_list = [], []
# # #     for i in range(len(x) - timesteps):
# # #         x_seq = x[i:i+timesteps]
# # #         y_seq = y[i+timesteps]
# # #         x_list.append(x_seq)
# # #         y_list.append(y_seq)
# # #     return np.array(x_list), np.array(y_list)

# # # x, y = split_xy(x_data, y_data, timesteps=timesteps)

# # # print('x.shape:', x.shape)  # x.shape: (420397, 24, 13)
# # # print('y.shape:', y.shape)  # y.shape: (420397,)


# # # x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=42)
    

# # # model = Sequential()
# # # model.add(GRU(64, input_shape=(x.shape[1], x.shape[2]), return_sequences=False))
# # # model.add(Dense(32, activation='relu'))
# # # model.add(Dense(32, activation='relu'))
# # # model.add(Dense(32, activation='relu'))
# # # model.add(Dense(1))  # 예측값은 wd (deg), 1개

# # # model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # or just omit metrics

# # # # model.summary()


# # # history = model.fit(x_train, y_train, epochs = 100, batch_size = 64, verbose = 1, validation_split=0.2)


# # # loss, acc = model.evaluate(x_test, y_test)

# # # y_pred = model.predict(x_test)
# # # rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# # print('loss (MSE):', loss)
# # print('acc:', acc)
# # print('RMSE:', rmse)
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from keras.models import Sequential
# from keras.layers import GRU, Dense, Dropout, BatchNormalization,LSTM
# from keras.callbacks import ModelCheckpoint
 
# #  1. 데이터 불러오기
# file_path = '/workspace/TensorJae/Study25/_data/jena_climate_2009_2016.csv'
# df = pd.read_csv(file_path, parse_dates=['Date Time'])

# # 2. 예측 구간 정의 및 분리
# start_pred = pd.to_datetime("31.12.2016 00:10:00", dayfirst=True)
# end_pred = pd.to_datetime("01.01.2017 00:00:00", dayfirst=True)

# df_train = df[~((df['Date Time'] >= start_pred) & (df['Date Time'] <= end_pred))].copy()
# df_pred = df[(df['Date Time'] >= start_pred) & (df['Date Time'] <= end_pred)].copy()

# # 3. 전처리 대상 컬럼 정의
# minmax_cols = ['rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)']
# exclude_cols = ['Date Time', 'wd (deg)'] + minmax_cols
# standard_cols = [col for col in df_train.columns if col not in exclude_cols]
# feature_columns = standard_cols + minmax_cols
# target_column = 'wd (deg)'

# # 4. 정규화 (학습용)
# scaler_minmax = MinMaxScaler().fit(df_train[minmax_cols])
# scaler_standard = StandardScaler().fit(df_train[standard_cols])

# df_train_scaled = df_train.copy()
# df_train_scaled[minmax_cols] = scaler_minmax.transform(df_train[minmax_cols])
# df_train_scaled[standard_cols] = scaler_standard.transform(df_train[standard_cols])

# # 5. x, y 데이터 생성 (wd 제외)
# x_data = df_train_scaled[feature_columns].values
# y_data = df_train[target_column].values

# # 6. 슬라이딩 윈도우
# def split_xy(x, y, timesteps):
#     x_list, y_list = [], []
#     for i in range(len(x) - timesteps):
#         x_seq = x[i:i+timesteps]
#         y_seq = y[i+timesteps]
#         x_list.append(x_seq)
#         y_list.append(y_seq)
#     return np.array(x_list), np.array(y_list)

# timesteps = 24
# x, y = split_xy(x_data, y_data, timesteps=timesteps)

# # 7. 학습/검증 분리
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

# # 8. 모델 구성
# model = Sequential([
#     LSTM(64, input_shape=(x.shape[1], x.shape[2]), return_sequences=False),
#     BatchNormalization(),
#     Dense(64, activation='relu'),
#     Dropout(0.2),
#     BatchNormalization(),
#     Dense(32, activation='relu'),
#     Dropout(0.2),
#     Dense(32, activation='relu'),
#     Dense(1)
# ])
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# # 9. 학습
# model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=1)

# # 10. 예측 대상 데이터 전처리 (정답은 사용하지 않음)
# df_pred_scaled = df_pred.copy()
# df_pred_scaled[minmax_cols] = scaler_minmax.transform(df_pred_scaled[minmax_cols])
# df_pred_scaled[standard_cols] = scaler_standard.transform(df_pred_scaled[standard_cols])
# x_pred_full = df_pred_scaled[feature_columns].values

# # 슬라이딩 윈도우로 121개 샘플 생성
# x_pred = []
# for i in range(len(x_pred_full) - timesteps):
#     x_pred.append(x_pred_full[i:i+timesteps])
# x_pred = np.array(x_pred)

# # 예측
# y_pred = model.predict(x_pred).flatten()

# # datetime 매칭
# submission_datetimes = df_pred.iloc[timesteps:]['Date Time'].reset_index(drop=True)

# # submission 저장
# submission_df = pd.DataFrame({
#     'datetime': submission_datetimes,
#     'wd': y_pred
# })
# submission_df.to_csv('/workspace/TensorJae/Study25/_save/jena/submission.csv', index=False)
# print("✅ submission.csv 저장 완료:", submission_df.shape)

# # # ✅ 최종 성능:
# # # Loss (MSE): 7159.671875
# # # MAE: 71.4510498046875
# # # RMSE: 84.61482697638445
# # # 4/4 [==============================] - 0s 4ms/step
# # # ✅ submission.csv 저장 완료: (120, 2)


# # import pandas as pd
# # import numpy as np
# # import os
# # from sklearn.preprocessing import MinMaxScaler, StandardScaler
# # from keras.models import Sequential, load_model
# # from keras.layers import LSTM, Dense, Dropout, BatchNormalization, GRU
# # from keras.callbacks import EarlyStopping, ModelCheckpoint

# # # --- 1. Configuration and Path Setup ---
# # FILE_PATH = '/workspace/TensorJae/Study25/_data/jena_climate_2009_2016.csv'
# # SAVE_DIR = '/workspace/TensorJae/Study25/_save/jena/'
# # os.makedirs(SAVE_DIR, exist_ok=True)

# # MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'best_jena_model.h5')
# # SUBMISSION_PATH = os.path.join(SAVE_DIR, 'submission.csv')

# # TIMESTEPS = 24 * 6 # Use 24 hours of data (since data is every 10 mins, 24*6=144)
# # EPOCHS = 5
# # BATCH_SIZE = 64

# # # --- 2. Data Loading and Splitting ---
# # print("✅ 1. Loading and splitting data...")
# # df = pd.read_csv(FILE_PATH, parse_dates=['Date Time'])

# # # Define the exact period to be predicted
# # start_pred = pd.to_datetime("31.12.2016 00:10:00", dayfirst=True)
# # end_pred = pd.to_datetime("01.01.2017 00:00:00", dayfirst=True)

# # # Separate the dataset into training/validation and the final prediction set
# # df_train_val = df[df['Date Time'] < start_pred].copy()
# # df_to_predict = df[(df['Date Time'] >= start_pred) & (df['Date Time'] <= end_pred)].copy()

# # # --- 3. Feature Engineering (Handling Cyclical Wind Direction) ---
# # print("✅ 2. Performing feature engineering for cyclical data...")
# # target_column = 'wd (deg)'
# # # Convert degrees to radians for trigonometric functions
# # wd_rad = df_train_val[target_column] * np.pi / 180
# # # Decompose the angle into x and y components
# # df_train_val['wd_x'] = np.cos(wd_rad)
# # df_train_val['wd_y'] = np.sin(wd_rad)

# # # --- 4. Preprocessing and Scaling ---
# # print("✅ 3. Preprocessing and scaling data...")
# # # Define columns for different scaling strategies
# # minmax_cols = ['rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)']
# # # Exclude original time, original target, and new engineered targets
# # exclude_cols = ['Date Time', 'wd (deg)', 'wd_x', 'wd_y'] + minmax_cols
# # standard_cols = [col for col in df_train_val.columns if col not in exclude_cols]
# # feature_columns = standard_cols + minmax_cols

# # # Fit scalers ONLY on the training/validation data
# # scaler_minmax = MinMaxScaler().fit(df_train_val[minmax_cols])
# # scaler_standard = StandardScaler().fit(df_train_val[standard_cols])

# # # Apply scaling to the training/validation data
# # df_train_val_scaled = df_train_val.copy()
# # df_train_val_scaled[minmax_cols] = scaler_minmax.transform(df_train_val[minmax_cols])
# # df_train_val_scaled[standard_cols] = scaler_standard.transform(df_train_val[standard_cols])

# # # --- 5. Create Windowed Dataset ---
# # def create_windowed_dataset(x_data, y_data, timesteps):
# #     x_list, y_list = [], []
# #     for i in range(len(x_data) - timesteps):
# #         x_list.append(x_data[i : i + timesteps])
# #         y_list.append(y_data[i + timesteps])
# #     return np.array(x_list), np.array(y_list)

# # x_data = df_train_val_scaled[feature_columns].values
# # y_data = df_train_val[['wd_x', 'wd_y']].values

# # x, y = create_windowed_dataset(x_data, y_data, timesteps=TIMESTEPS)

# # # --- 6. Chronological Train/Validation Split ---
# # print("✅ 4. Splitting data chronologically...")
# # split_index = int(len(x) * 0.9) # 90% for training, 10% for validation
# # x_train, x_val = x[:split_index], x[split_index:]
# # y_train, y_val = y[:split_index], y[split_index:]

# # print(f"Train shapes: x={x_train.shape}, y={y_train.shape}")
# # print(f"Validation shapes: x={x_val.shape}, y={y_val.shape}")

# # # --- 7. Model Building ---
# # print("✅ 5. Building the model...")
# # model = Sequential([
# #     GRU(128, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True),
# #     Dropout(0.3),
# #     GRU(64, return_sequences=False),
# #     Dropout(0.3),
# #     Dense(64, activation='relu'),
# #     BatchNormalization(),
# #     Dense(2) # Output layer predicts 2 values: wd_x and wd_y
# # ])

# # model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# # model.summary()

# # # --- 8. Model Training ---
# # print("✅ 6. Starting model training...")
# # es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# # mcp = ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)

# # history = model.fit(
# #     x_train, y_train,
# #     epochs=EPOCHS,
# #     batch_size=BATCH_SIZE,
# #     validation_data=(x_val, y_val),
# #     callbacks=[es, mcp],
# #     verbose=1
# # )

# # # ====================================================================================
# # # ✅✅✅ NEW: 9. Model Evaluation ✅✅✅
# # # ====================================================================================
# # print("\n✅ 7. Evaluating model performance on validation set...")
# # best_model = load_model(MODEL_SAVE_PATH)

# # # A. Get the loss (MSE) and MAE on the (x,y) coordinates
# # loss, mae = best_model.evaluate(x_val, y_val, verbose=0)
# # print("--- Model Performance ---")
# # print(f"Loss (MSE on coordinates): {loss:.4f}")
# # print(f"MAE (on coordinates): {mae:.4f}")

# # # B. Calculate metrics in interpretable degrees
# # # Predict the (x,y) coordinates for the validation set
# # y_pred_val_xy = best_model.predict(x_val)

# # # Convert both true and predicted coordinates back to degrees
# # true_rad = np.arctan2(y_val[:, 1], y_val[:, 0])
# # pred_rad = np.arctan2(y_pred_val_xy[:, 1], y_pred_val_xy[:, 0])

# # true_deg = np.degrees(true_rad)
# # pred_deg = np.degrees(pred_rad)

# # true_deg = np.where(true_deg < 0, true_deg + 360, true_deg)
# # pred_deg = np.where(pred_deg < 0, pred_deg + 360, pred_deg)

# # # Calculate the shortest angle difference (handles 359 -> 1 degree wrap-around)
# # angular_diff = 180 - np.abs(np.abs(true_deg - pred_deg) - 180)

# # # Calculate RMSE in degrees
# # rmse_deg = np.sqrt(np.mean(angular_diff**2))
# # print(f"RMSE (in degrees): {rmse_deg:.4f}")

# # # Calculate custom "accuracy" (e.g., % of predictions within ±15 degrees)
# # TOLERANCE_DEGREES = 15.0
# # accuracy_custom = np.mean(angular_diff <= TOLERANCE_DEGREES) * 100
# # print(f"Accuracy (within ±{TOLERANCE_DEGREES}°): {accuracy_custom:.2f}%")
# # print("-------------------------\n")


# # # --- 10. Prediction for Submission ---
# # print("✅ 8. Preparing data and making final predictions...")
# # # To predict the first value of the target period, we need the last `TIMESTEPS` data points from the training set.
# # last_train_data = df_train_val.tail(TIMESTEPS)
# # prediction_input_df = pd.concat([last_train_data, df_to_predict], ignore_index=True)

# # # Scale the combined dataframe for prediction
# # prediction_input_scaled = prediction_input_df.copy()
# # prediction_input_scaled[minmax_cols] = scaler_minmax.transform(prediction_input_df[minmax_cols])
# # prediction_input_scaled[standard_cols] = scaler_standard.transform(prediction_input_df[standard_cols])

# # # Create windowed data for prediction
# # x_pred_data = prediction_input_scaled[feature_columns].values
# # x_to_predict = []
# # for i in range(len(x_pred_data) - TIMESTEPS):
# #     x_to_predict.append(x_pred_data[i : i + TIMESTEPS])
# # x_to_predict = np.array(x_to_predict)

# # # Make predictions
# # y_pred_xy = best_model.predict(x_to_predict)

# # # --- 11. Post-processing and Submission File Creation ---
# # print("✅ 9. Post-processing predictions and saving submission file...")
# # # Convert predicted (x,y) coordinates back to degrees
# # y_pred_rad = np.arctan2(y_pred_xy[:, 1], y_pred_xy[:, 0]) # Note: arctan2(y, x)
# # y_pred_deg = np.degrees(y_pred_rad)
# # # Ensure degrees are in the [0, 360] range
# # y_pred_deg = np.where(y_pred_deg < 0, y_pred_deg + 360, y_pred_deg)

# # # Create the submission dataframe
# # submission_df = pd.DataFrame({
# #     'Date Time': df_to_predict['Date Time'],
# #     'wd (deg)': y_pred_deg
# # })

# # submission_df.to_csv(SUBMISSION_PATH, index=False)
# # print(f"✅ Submission file saved to: {SUBMISSION_PATH}")
# # print(submission_df.head())

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# 1. 경로 설정
file_path = '/workspace/TensorJae/Study25/_data/jena_climate_2009_2016.csv'
save_path = '/workspace/TensorJae/Study25/_save/jena/best_model.h5'
submission_path = '/workspace/TensorJae/Study25/_save/jena/submission.csv'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 2. 데이터 로딩
df = pd.read_csv(file_path, parse_dates=['Date Time'])

# 3. 예측 대상 기간 분리
start_pred = pd.to_datetime("31.12.2016 00:10:00", dayfirst=True)
end_pred = pd.to_datetime("01.01.2017 00:00:00", dayfirst=True)
df_train = df[~((df['Date Time'] >= start_pred) & (df['Date Time'] <= end_pred))].copy()
df_pred = df[(df['Date Time'] >= start_pred) & (df['Date Time'] <= end_pred)].copy()

# 4. 정규화 대상 컬럼 정의
minmax_cols = ['rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)']
exclude_cols = ['Date Time', 'wd (deg)'] + minmax_cols
standard_cols = [col for col in df_train.columns if col not in exclude_cols]
feature_columns = standard_cols + minmax_cols
target_column = 'wd (deg)'

# 5. 정규화
scaler_minmax = MinMaxScaler().fit(df_train[minmax_cols])
scaler_standard = StandardScaler().fit(df_train[standard_cols])
df_train_scaled = df_train.copy()
df_train_scaled[minmax_cols] = scaler_minmax.transform(df_train[minmax_cols])
df_train_scaled[standard_cols] = scaler_standard.transform(df_train[standard_cols])

# 6. 슬라이딩 윈도우 함수
def split_xy(x, y, timesteps):
    x_list, y_list = [], []
    for i in range(len(x) - timesteps):
        x_seq = x[i:i+timesteps]
        y_seq = y[i+timesteps]
        x_list.append(x_seq)
        y_list.append(y_seq)
    return np.array(x_list), np.array(y_list)

timesteps = 24
x_data = df_train_scaled[feature_columns].values
y_data = df_train[target_column].values
x, y = split_xy(x_data, y_data, timesteps=timesteps)

# 7. 시계열 기반 학습/검증 분리
split_idx = int(len(x) * 0.8)
x_train, x_val = x[:split_idx], x[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# 8. 모델 구성
model = Sequential([
    LSTM(64, input_shape=(x.shape[1], x.shape[2]), return_sequences=False),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 9. 모델 저장 콜백
checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1)

# 10. 모델 학습
history = model.fit(
    x_train, y_train,
    epochs=1,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint],
    verbose=1
)

# 11. 모델 로드 및 평가
best_model = load_model(save_path)
y_val_pred = best_model.predict(x_val).flatten()

rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae = mean_absolute_error(y_val, y_val_pred)

print(f"\n✅ 검증 RMSE: {rmse:.4f}")
print(f"✅ 검증 MAE: {mae:.4f}")

# 12. 예측용 데이터 전처리
df_pred_scaled = df_pred.copy()
df_pred_scaled[minmax_cols] = scaler_minmax.transform(df_pred_scaled[minmax_cols])
df_pred_scaled[standard_cols] = scaler_standard.transform(df_pred_scaled[standard_cols])
x_pred_full = df_pred_scaled[feature_columns].values

# 슬라이딩 윈도우 생성
x_pred = []
for i in range(len(x_pred_full) - timesteps):
    x_pred.append(x_pred_full[i:i+timesteps])
x_pred = np.array(x_pred)

# 예측
y_pred = best_model.predict(x_pred).flatten()
submission_datetimes = df_pred.iloc[timesteps:]['Date Time'].reset_index(drop=True)

submission_df = pd.DataFrame({
    'datetime': submission_datetimes,
    'wd': y_pred
})
submission_df.to_csv(submission_path, index=False)
print("✅ submission.csv 저장 완료:", submission_df.shape)
