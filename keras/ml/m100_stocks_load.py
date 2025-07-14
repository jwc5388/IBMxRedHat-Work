import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load

# 경로
if os.path.exists('/workspace/TensorJae/Study25/'):
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')

path = os.path.join(BASE_PATH, '_data/stocks/')
save_path = os.path.join(path, '_save/')


#데이터 
samsung_raw = pd.read_csv(path + 'samsung.csv', thousands=',')
hanwha_raw = pd.read_csv(path + 'hanwha.csv', thousands=',')
hanwha_raw = hanwha_raw.rename(columns={'Unnamed: 6': '대비_금액'})

# 액면분할 보정 함수
def cut(df):
    df = df.copy(); df['일자'] = pd.to_datetime(df['일자'], dayfirst=True)
    df = df.sort_values(by='일자', ascending=False).dropna()
    split_date = pd.to_datetime('2018-05-04')
    price_cols = ['시가', '고가', '저가', '종가']
    df.loc[df['일자'] < split_date, price_cols] /= 50
    return df

samsung_cut = cut(samsung_raw)
hanwha_cut = cut(hanwha_raw)

def get_latest_data(df, scaler):
    df = df.copy()
    df['일자'] = pd.to_datetime(df['일자'], dayfirst=True)
    df = df.sort_values(by='일자', ascending=False).dropna()
    df['시가_5일_MA'] = df['시가'].rolling(window=5).mean()
    df['시가_변화'] = df['시가'].diff()
    df['등락률_1일전'] = df['시가'].pct_change() * 100
    df = df.dropna()
    X = df.drop(columns=['일자', '대비'])
    latest_x = X.iloc[0]
    latest_x_scaled = scaler.transform([latest_x.values])
    return latest_x_scaled


print('loading saved scalers')

model_h_loaded = load(os.path.join(save_path, 'model_h.joblib'))
model_residual_loaded = load(os.path.join(save_path, 'model_residual.joblib'))
scaler_h_loaded = load(os.path.join(save_path, 'scaler_h.joblib'))
scaler_s_loaded = load(os.path.join(save_path, 'scaler_s.joblib'))

xh_latest = get_latest_data(hanwha_cut, scaler_h_loaded)
xs_latest = get_latest_data(samsung_cut, scaler_s_loaded)

pred_h_latest = model_h_loaded.predict(xh_latest)[0]
residual_latest = model_residual_loaded.predict(xs_latest)[0]

final_pred = pred_h_latest + residual_latest

print("\n📈 (가중치 로드) 2025년 7월 14일 시가 예측")
print(f" - 한화 단독 예측     : {int(pred_h_latest):,} 원")
print(f" - 삼성 기반 잔차 보정: {int(residual_latest):,} 원")
print(f" ✅ 최종 보정 예측     : {int(final_pred):,} 원")