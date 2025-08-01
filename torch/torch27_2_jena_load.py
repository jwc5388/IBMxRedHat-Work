# === inference.py ===
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from pathlib import Path

# -----------------------
# 0) Device
# -----------------------
DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

# -----------------------
# 1) 경로 설정
# -----------------------
# 네 환경에 맞게 수정
DATA_DIR = Path('/Users/jaewoo000/Desktop/IBM:RedHat/Study25/_data/kaggle/netflix/')
SAVE_DIR = Path('/Users/jaewoo000/Desktop/IBM:RedHat/Study25/_save/torch/')
WEIGHT_PATH = SAVE_DIR / 't25_netflix.pth'   # 저장해둔 가중치 파일

TRAIN_CSV = DATA_DIR / 'train.csv'
TEST_CSV  = DATA_DIR / 'test.csv'
SUB_CSV   = DATA_DIR / 'sample_submission.csv'  # 있으면 제출 형식으로 저장 시 활용

# -----------------------
# 2) Dataset (훈련 기준 min/max로 정규화)
# -----------------------
class SlidingDataset(Dataset):
    """
    df: DataFrame (train or test)
    x_cols: 입력 열 인덱스 목록 또는 컬럼명 목록 (여기서는 df.iloc[:, 1:4] 사용 가정)
    y_col: 타깃 컬럼명 (train 전용). test에는 None 가능.
    timesteps: 시계열 윈도 길이
    x_min, x_max: 훈련셋에서 구한 열별 min/max (정규화 기준)
    """
    def __init__(self, df, x_cols, y_col=None, timesteps=30, x_min=None, x_max=None):
        self.timesteps = timesteps

        if isinstance(x_cols, slice):
            X = df.iloc[:, x_cols].values.astype(np.float32)
        else:
            X = df.loc[:, x_cols].values.astype(np.float32)

        if x_min is None or x_max is None:
            # 훈련셋용: 내부에서 fit
            self.x_min = X.min(axis=0)
            self.x_max = X.max(axis=0)
        else:
            # 테스트셋용: 훈련셋 기준을 외부에서 주입
            self.x_min = x_min
            self.x_max = x_max

        denom = (self.x_max - self.x_min)
        denom[denom == 0.0] = 1.0  # 분모 0 방어
        X = (X - self.x_min) / denom

        self.X = X
        self.y = None
        if y_col is not None and y_col in df.columns:
            self.y = df[y_col].values.astype(np.float32)

        # 유효 길이
        self.N = len(self.X) - self.timesteps
        if self.N < 1:
            raise ValueError(f"행 개수({len(self.X)})가 timesteps({self.timesteps})보다 작아 슬라이딩 윈도우를 만들 수 없음.")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.timesteps]  # (T, F)
        x = torch.from_numpy(x)
        if self.y is None:
            return x, None
        y = self.y[idx + self.timesteps]        # 다음 시점
        return x, torch.tensor(y, dtype=torch.float32)

# -----------------------
# 3) 모델 정의 (훈련 시와 동일 구조)
# -----------------------
class RNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=3):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.rnn(x)     # (B, T, H)
        x = x[:, -1, :]        # (B, H)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1)    # (B,)

# -----------------------
# 4) 로드 + 평가/예측
# -----------------------
def main():
    timesteps = 30
    batch_size = 64

    # 4-1) 데이터 로드
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    # 입력 특성: 열 1~3 (예: Open, High, Low) 가정
    x_cols = slice(1, 4)
    y_col  = 'Close'  # 타깃

    # 4-2) 훈련셋 기준 정규화 파라미터 추출
    tmp_train = SlidingDataset(train_df, x_cols=x_cols, y_col=y_col, timesteps=timesteps)
    x_min, x_max = tmp_train.x_min.copy(), tmp_train.x_max.copy()

    # 정식 Dataset/DataLoader
    train_ds = SlidingDataset(train_df, x_cols=x_cols, y_col=y_col, timesteps=timesteps,
                              x_min=x_min, x_max=x_max)
    test_ds  = SlidingDataset(test_df,  x_cols=x_cols, y_col=None,   timesteps=timesteps,
                              x_min=x_min, x_max=x_max)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    # 4-3) 모델 로드
    model = RNN(input_size=3, hidden_size=64, num_layers=3).to(DEVICE)
    state = torch.load(WEIGHT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # 4-4) 훈련셋 평가(MSE, R²)
    criterion = nn.MSELoss()
    total_loss = 0.0
    y_pred_all, y_true_all = [], []

    with torch.no_grad():
        for xb, yb in train_loader:
            xb = xb.to(DEVICE).float()
            yb = yb.to(DEVICE).float()
            pred = model(xb)
            total_loss += criterion(pred, yb).item()
            y_pred_all.append(pred.cpu().numpy())
            y_true_all.append(yb.cpu().numpy())

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    r2 = r2_score(y_true, y_pred)

    print(f'✅ Train 평가 - 평균 MSE: {total_loss / len(train_loader):.6f}')
    print(f'✅ Train 평가 - R²: {r2:.6f}')

    # 4-5) 테스트셋 예측
    test_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE).float()
            pred = model(xb)
            test_preds.append(pred.cpu().numpy())
    test_preds = np.concatenate(test_preds).reshape(-1)

    # 4-6) 저장: sample_submission과 길이 맞추기
    # 슬라이딩 윈도우 특성상 예측 개수 = len(test) - timesteps
    # sample_submission 길이와 다를 수 있으므로 맞춰서 저장
    sub_path = DATA_DIR / 'sample_submission.csv'
    if sub_path.exists():
        sub = pd.read_csv(sub_path).copy()
        # 길이 조정 로직
        if len(test_preds) >= len(sub):
            sub['Close'] = test_preds[:len(sub)]
        else:
            # 부족하면 뒤를 마지막 값으로 채움(또는 NaN 채움)
            pad = np.full(len(sub) - len(test_preds), test_preds[-1] if len(test_preds) > 0 else 0.0)
            sub['Close'] = np.concatenate([test_preds, pad])
        out_path = SAVE_DIR / 'submission_inference.csv'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_path, index=False)
        print(f'📁 제출 파일 저장: {out_path}  (shape={sub.shape})')
    else:
        # 샘플 제출 파일이 없다면, 예측만 별도 저장
        out_path = SAVE_DIR / 'test_predictions_sliding.csv'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'pred': test_preds}).to_csv(out_path, index=False)
        print(f'📁 테스트 예측 저장: {out_path}  (len={len(test_preds)})')

if __name__ == "__main__":
    main()