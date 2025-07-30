from torchvision.datasets import MNIST
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from torch.utils.data import TensorDataset, DataLoader
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
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

path = basepath + '_data/torch/'


transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5),0.5)])

train_dataset = MNIST(path, train=True, download=True, transform=transf)
test_dataset = MNIST(path, train=False, download=True, transform=transf
)

print(len(train_dataset))

img_tensor, label = train_dataset[0]

print(label)
print(img_tensor.shape)
print(img_tensor.min(), img_tensor.max())


train_loader = DataLoader(train_dataset, batch_size=32, shuffle= False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle= False)

class CNN(nn.Module):
    def __init__(self, num_features):
        super(CNN, self).__init__()

        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size= (3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout(0.2)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 32,  kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(32,16, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2)
        )

        self.flatten = nn.Flatten()

        self.hidden_layer4 = nn.Sequential(
            nn.Linear(16*5*5, 64),
            nn.ReLU()
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(32,10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.flatten(x)
        # x = x.view(x.shape[0], -1)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        
        return x
    
model = CNN(1).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        hypothesis = model(x_batch)

        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()

        y_predict = torch.argmax(hypothesis,1)
        # acc = 

