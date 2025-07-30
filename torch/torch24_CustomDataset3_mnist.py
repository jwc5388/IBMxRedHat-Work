#mnist 로 커스텀데이터셋 맹그러봐!!

from torchvision.datasets import MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.metrics import accuracy_score, r2_score
import torchvision.transforms as tr


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



transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5),(0.5))]) #크기 늘리고 to tensor

# train_dataset = MNIST(path, train=True, download=True, transform=transf)
# test_dataset = MNIST(path, train=False, download=True, transform=transf)

class MyDataset(Dataset):
    def __init__(self):
        self.mnist = MNIST(path, train= True, download=True, transform=transf )

    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, index):
        return self.mnist[index]

dataset = MyDataset()

loader = DataLoader(dataset, batch_size=32, shuffle=True)

#4 출력
for batch_idx, (xb, yb) in enumerate(loader):
    print('======================== 배치:', batch_idx, '=================================')
    print('x: 배치', batch_idx)
    print('x:', xb)
    print('y:', yb
          )