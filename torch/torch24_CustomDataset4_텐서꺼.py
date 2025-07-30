#mnist 로 커스텀데이터셋 맹그러봐!!

from torchvision.datasets import MNIST
from keras.datasets import mnist
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.metrics import accuracy_score, r2_score
import torchvision.transforms as tr
from PIL import Image


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
    
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)


SEED = 42
import random
random.seed(SEED) #python random fix
np.random.seed(SEED) #numpy random fix
torch.manual_seed(SEED) #torch random fix
torch.cuda.manual_seed(SEED) #torch cuda random fix


import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

path = basepath + '_data/torch/'



# transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5),(0.5))]) #크기 늘리고 to tensor

# train_dataset = MNIST(path, train=True, download=True, transform=transf)
# test_dataset = MNIST(path, train=False, download=True, transform=transf)

class MyDataset(Dataset):
    def __init__(self, is_train = True):
        (x_train, y_train), (x_test,y_test) = mnist.load_data()
        if is_train:
            self.images = x_train
            self.labels = y_train
        else:
            self.images = x_test
            self.labels = y_test


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]                     # numpy (28,28)
        label = int(self.labels[idx])              # 정수 라벨
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1,28,28)
        img_tensor = (img_tensor / 255.0 - 0.5) / 0.5                     # 정규화
        return img_tensor, label
    

train_dataset = MyDataset(is_train=True)
test_dataset = MyDataset(is_train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ✅ 출력 확인
for batch_idx, (xb, yb) in enumerate(train_loader):
    print(f'================ 배치: {batch_idx} ==================')
    print('x.shape:', xb.shape)  # (32, 1, 56, 56)
    print('y:', yb)
    if batch_idx == 10:
        break