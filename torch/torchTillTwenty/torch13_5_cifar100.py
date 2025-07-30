#실습!!!!
#드랍아웃 적용해보아요

from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST, CIFAR10, CIFAR100
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, r2_score


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

train_dataset = CIFAR100(path, train=True, download=True)
test_dataset = CIFAR100(path, train=False, download=True)

print(train_dataset)

print(type(train_dataset))
print(train_dataset[0])


x_train, y_train = train_dataset.data/255., train_dataset.targets
x_test, y_test = test_dataset.data/255., test_dataset.targets

print(x_train)
print(y_train)


print(x_train.shape, y_train.size()) 


print(np.min(x_train.numpy()), np.max(x_train.numpy()))


exit()


x_train, x_test = x_train.view(-1, 28*28), x_test.reshape(-1, 784)

print(x_train.shape, x_test.size())


print("x_test.shape:", x_test.shape)
print("y_test.shape:", y_test.shape)

# exit()

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)



train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
# exit()

class DNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # super(self, DNN).__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
            
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(64, 64),
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
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        
        return x


model = DNN(784).to(DEVICE)

criterion= nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1e-4)

def train(model, criterion, optimizer, loader):
    #model.train()
    epoch_loss = 0
    epoch_acc = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        
        loss= criterion(hypothesis, y_batch)
        
        
        loss.backward()
        optimizer.step() # w = x -lr*로스를 웨이트로 미분한값 
        
        y_predict = torch.argmax(hypothesis,1)
        acc = (y_predict == y_batch).float().mean()
        
        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss/len(loader), epoch_acc / len(loader)



def evaluate(model, criterion, loader):
    model.eval()
    #model.train()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            
            loss= criterion(hypothesis, y_batch)
            
            
            y_predict = torch.argmax(hypothesis,1)
            acc = (y_predict == y_batch).float().mean()
            
            epoch_loss += loss.item()
            epoch_acc += acc

        return epoch_loss/len(loader), epoch_acc / len(loader)



epochs = 10

for epoch in range(1, epochs+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    print(f'epoch : {epoch}, loss : {loss:.4f}, acc : {acc:.3f}, \
          val_loss:{val_loss:.4f}, val_acc:{val_acc:.3f}')
    
loss, acc = evaluate(model, criterion, test_loader)
print('======================================')


print('최종 loss:', loss)
print('최종 acc:', acc.item())




# 최종 loss: 0.4915915798788634
# 최종 acc: 0.8476437926292419
    
        
# def evaluate(model, criterion, loader):
#     model.eval()
#     total_loss = 0
    
#     for x_batch, y_batch in loader:
#         with torch.no_grad():
#             y_pred = model(x_batch)
#             loss2 = criterion(y_pred, y_batch)
            
#             total_loss += loss2.item()
            
#         return total_loss/len(loader)
    
# last_loss = evaluate(model, criterion, test_loader)
        
# print('최종 loss:', last_loss)

# y_predict = model(x_test)

# y_pred = np.round(y_predict.detach().cpu().numpy())
# y_true = y_test.detach().cpu().numpy()

# accuracy = accuracy_score(y_true, y_pred)
# r2 = r2_score(y_true, y_pred)
# # acc = accuracy_score(y_test.detach().cpu().numpy(), y_predict.detach().cpu().numpy().round())

# print('accuracy score:', accuracy)