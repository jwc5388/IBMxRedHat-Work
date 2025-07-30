# # # # # # # # # # # import torch
# # # # # # # # # # # import torch.version


# # # # # # # # # # # import numpy as np
# # # # # # # # # # # import torch 
# # # # # # # # # # # import torch.nn as nn
# # # # # # # # # # # import torch.optim as optim

# # # # # # # # # # # if torch.cuda.is_available():
# # # # # # # # # # #     DEVICE = torch.device('cuda')
# # # # # # # # # # # elif torch.backends.mps.is_available():
# # # # # # # # # # #     DEVICE = torch.device('mps')
# # # # # # # # # # # else:
# # # # # # # # # # #     DEVICE = torch.device('cpu')
    
# # # # # # # # # # # print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

# # # # # # # # # # # x = np.array([1,2,3])
# # # # # # # # # # # y = np.array([1,2,3])

# # # # # # # # # # # x = torch.FloatTensor(x)
# # # # # # # # # # # x = x.reshape(-1,1)
# # # # # # # # # # # y = torch.FloatTensor(y).unsqueeze(1)

# # # # # # # # # # # model = nn.Linear(1,1) #input output

# # # # # # # # # # # criterion = nn.MSELoss()
# # # # # # # # # # # optimizer = optim.SGD(model.parameters(), lr = 0.1)

# # # # # # # # # # # def train(model, criterion, optimizer, x, y):
# # # # # # # # # # #     optimizer.zero_grad()#기울기 초기화 각 배치마다 기울기를 초기화하여, 기울기 누적에 의한 문제 해결
# # # # # # # # # # #     hypothesis = model(x)   #y = xw+b
# # # # # # # # # # #     loss = criterion(hypothesis, y) # loss = mse() = 시그마 (y-hypothesis)^2/n
# # # # # # # # # # #     # 	•	모델에 x를 넣어서 예측값을 얻습니다.
# # # # # # # # # # # 	# •	이 예측값을 가설(hypothesis) 이라고 부릅니다.
    
# # # # # # # # # # #     loss.backward() #기울기 값까지만 계산
# # # # # # # # # # #     optimizer.step()    # 가중치 갱신
# # # # # # # # # # #     return loss.item()

# # # # # # # # # # # epochs = 700
# # # # # # # # # # # for epoch in range(1, epochs+1):
# # # # # # # # # # #     loss = train(model, criterion, optimizer, x, y)
# # # # # # # # # # #     print('epoch: {}, loss: {}'.format(epoch, loss))    

# # # # # # # # # # # def evaluate(model, criterion, x, y):
# # # # # # # # # # #     model.eval()
# # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # #         y_predict = model(x)
# # # # # # # # # # #         loss2 = criterion(y, y_predict)
# # # # # # # # # # #     return loss2.item()

# # # # # # # # # # # loss2 = evaluate(model, criterion, x, y)
# # # # # # # # # # # print('최종 loss:', loss2)

# # # # # # # # # # # result = model(torch.Tensor([[4]]))
# # # # # # # # # # # print('4의 예측값:', result)
# # # # # # # # # # # print('4의 예측값', result.item())


# # # # # # # # # # import numpy as np
# # # # # # # # # # import torch
# # # # # # # # # # import torch.nn as nn
# # # # # # # # # # import torch.optim as optim

# # # # # # # # # # x = np.array([1,2,3])
# # # # # # # # # # y = np.array([1,2,3])

# # # # # # # # # # x = torch.FloatTensor(x)


# # # # # # # # # # x = torch.FloatTensor(x).unsqueeze(1)

# # # # # # # # # # y = torch.FloatTensor(y).unsqueeze(1)

# # # # # # # # # # model = nn.Linear(1,1)

# # # # # # # # # # criterion = nn.MSELoss()

# # # # # # # # # # optimizer = optim.SGD(model.parameters(), lr=0.1)

# # # # # # # # # # def train(model, criterion, optimizer, x, y):
# # # # # # # # # #     optimizer.zero_grad()
# # # # # # # # # #     hypothesis = model(x)
# # # # # # # # # #     loss = criterion(hypothesis, y)
# # # # # # # # # #     loss.backward()
# # # # # # # # # #     optimizer.step()
# # # # # # # # # #     return loss.item()


# # # # # # # # # # epochs = 1000,
# # # # # # # # # # for epoch in range(1,epochs+1):
# # # # # # # # # #     loss = train(model, criterion, optimizer, x,y,)
# # # # # # # # # #     print('epoch:{}, loss:{}'.format(epoch, loss))
    
# # # # # # # # # # def evaluate(model, criterion, x, y):
# # # # # # # # # #     model.eval()
# # # # # # # # # #     with torch.no_grad():
# # # # # # # # # #         y_predict = model(x)
# # # # # # # # # #         loss2 = criterion(y, y_predict)
        
# # # # # # # # # #     return loss2.item()


# # # # # # # # # # loss2 = evaluate(model, criterion, x,y)
# # # # # # # # # # print('최종 loss:', loss2)

# # # # # # # # # # result = model(torch.Tensor([[4]]))
# # # # # # # # # # print('4의 예측값:', result)
# # # # # # # # # # print('')

# # # # # # # # # import numpy as np
# # # # # # # # # import torch
# # # # # # # # # import torch.nn as nn
# # # # # # # # # import torch.optim as optim

# # # # # # # # # if torch.cuda.is_available():
# # # # # # # # #     DEVICE = torch.device('cuda')
# # # # # # # # # elif torch.backends.mps.is_available():
# # # # # # # # #     DEVICE = torch.device('mps')
# # # # # # # # # else:
# # # # # # # # #     DEVICE = torch.device('cpu')
    
# # # # # # # # # print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

# # # # # # # # # x = np.array([[1,2,3,4,5,6,7,8,9,10],[1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]]).transpose()
# # # # # # # # # y = np.array([1,2,3,4,5,6,7,7,9,10])


# # # # # # # # # x = torch.FloatTensor(x).to(DEVICE)

# # # # # # # # # y = y.reshape(-1,1)
# # # # # # # # # y = torch.FloatTensor(y).to(DEVICE)

# # # # # # # # # x_mean = torch.mean(x)
# # # # # # # # # x_std = torch.std(x)

# # # # # # # # # x = (x-torch.mean(x))/ torch.std(x)

# # # # # # # # # model = nn.Sequential(
# # # # # # # # #     nn.Linear(2, 5),
# # # # # # # # #     nn.Linear(5,4),
# # # # # # # # #     nn.Linear(4,3),
# # # # # # # # #     nn.Linear(3,2),
# # # # # # # # #     nn.Linear(2,1),
    
# # # # # # # # # ).to(DEVICE)


# # # # # # # # # criterion= nn.MSELoss()

# # # # # # # # # optimizer = optim.SGD(model.parameters(), lr= 0.01)

# # # # # # # # # def train(model, criterion, optimizer, x,y):
# # # # # # # # #     optimizer.zero_grad()
# # # # # # # # #     hypothesis = model(x)
# # # # # # # # #     loss = criterion(hypothesis, y)
# # # # # # # # #     loss.backward()
# # # # # # # # #     optimizer.step()
# # # # # # # # #     return loss.item()

# # # # # # # # # epochs = 1000
# # # # # # # # # for epoch in range(1, epochs+1):
# # # # # # # # #     loss = train(model, criterion,optimizer, x,y)
# # # # # # # # #     print('epoch:{}, loss:{}'.format(epoch, loss))

# # # # # # # # # def evaluate(model, criterion, x,y):
# # # # # # # # #     model.eval()
# # # # # # # # #     with torch.no_grad():
# # # # # # # # #         y_predict = model(x)
# # # # # # # # #         loss2 = criterion(y, y_predict)
# # # # # # # # #     return loss2.item()

# # # # # # # # # loss2 = evaluate(model, criterion, x,y) 
# # # # # # # # # print('최종 loss:', loss2)

# # # # # # # # # x_pred = (torch.Tensor([[10,1.3]]).to(DEVICE)- x_mean)/x_std
# # # # # # # # # result = model(x_pred)

# # # # # # # # # print('10, 1.3의 예측값:', result)
# # # # # # # # # print('10, 1.3의 예측값', result.item())   


# # # # # # # # import numpy as np
# # # # # # # # import torch
# # # # # # # # import torch.nn as nn
# # # # # # # # import torch.optim as optim
# # # # # # # # from sklearn.datasets import load_breast_cancer, fetch_california_housing
# # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # from sklearn.preprocessing import StandardScaler
# # # # # # # # from sklearn.metrics import accuracy_score
# # # # # # # # from sklearn.metrics import r2_score
# # # # # # # # import random

# # # # # # # # SEED = 42
# # # # # # # # import random
# # # # # # # # random.seed(SEED)
# # # # # # # # np.random.seed(SEED)
# # # # # # # # torch.manual_seed(SEED)
# # # # # # # # torch.cuda.manual_seed(SEED)


# # # # # # # # if torch.cuda.is_available():
# # # # # # # #     DEVICE= torch.device('cuda')
# # # # # # # # elif torch.backends.mps.is_available():
# # # # # # # #     DEVICE = torch.device('mps')
# # # # # # # # else:
# # # # # # # #     DEVICE = torch.device('cpu')


# # # # # # # # print('torch:', torch.__version__, '사용 device:', DEVICE)

# # # # # # # # dataset = fetch_california_housing()
# # # # # # # # x = dataset.data
# # # # # # # # y = dataset.target

# # # # # # # # x_train, x_test, y_train, y_test  = train_test_split(
# # # # # # # #     x,y, train_size=0.8, shuffle=True, random_state=SEED
# # # # # # # # )

# # # # # # # # scaler = StandardScaler()
# # # # # # # # x_train = scaler.fit_transform(x_train)
# # # # # # # # x_test = scaler.transform(x_test)

# # # # # # # # x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
# # # # # # # # x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
# # # # # # # # y_train = torch.tensor(y_train, dtype= torch.float32).unsqueeze(1).to(DEVICE)
# # # # # # # # y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

# # # # # # # # print(x_train.dtype)
# # # # # # # # print(x_train.shape, y_train.shape)
# # # # # # # # print(type())

# # # # # # # import numpy as np
# # # # # # # import torch
# # # # # # # import torch.nn as nn
# # # # # # # import torch.optim as optim
# # # # # # # from sklearn.datasets import load_breast_cancer
# # # # # # # from sklearn.preprocessing import StandardScaler
# # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # from sklearn.metrics import accuracy_score
# # # # # # # from sklearn.metrics import r2_score
# # # # # # # import random
# # # # # # # SEED = 42
# # # # # # # random.seed(SEED)
# # # # # # # np.random.seed(SEED)
# # # # # # # torch.manual_seed(SEED)

# # # # # # # if torch.cuda.is_available():
# # # # # # #     DEVICE = torch.device('cuda')
# # # # # # # elif torch.backends.mps.is_available():
# # # # # # #     DEVICE = torch.device('mps')
# # # # # # # else:
# # # # # # #     DEVICE = torch.device('cpu')
    
# # # # # # # print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)


# # # # # # # dataset = load_breast_cancer()
# # # # # # # x = dataset.data
# # # # # # # y = dataset.target

# # # # # # # print(x.shape)
# # # # # # # print(y.shape)

# # # # # # # x_train, x_test, y_train, y_test = train_test_split(
# # # # # # #     x, y, train_size=0.8, shuffle=True, random_state=SEED
# # # # # # # )

# # # # # # # scaler = StandardScaler()
# # # # # # # x_train = scaler.fit_transform(x_train)
# # # # # # # x_test = scaler.transform(x_test)

# # # # # # # x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
# # # # # # # x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
# # # # # # # y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
# # # # # # # y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)


# # # # # # # print(x_train.dtype)
# # # # # # # print(x_train.shape, y_train.shape)
# # # # # # # print(type(x_train))

# # # # # # # from torch.utils.data import TensorDataset
# # # # # # # from torch.utils.data import DataLoader

# # # # # # # train_set = TensorDataset(x_train, y_train)
# # # # # # # test_set = TensorDataset(x_test, y_test)

# # # # # # # print(train_set)
# # # # # # # print(type(train_set))
# # # # # # # print(len(train_set))
# # # # # # # print(train_set[0])

# # # # # # # print(train_set[0][0])
# # # # # # # print(train_set[0][1])

# # # # # # # train_loader = DataLoader(train_set, batch_size=100, shuffle=False)
# # # # # # # test_loader = DataLoader(test_set, batch_size=100, shuffle=False)


# # # # # # # class Model(nn.Module):
# # # # # # #     def __init__(self, input_dim, output_dim):
# # # # # # #         super(Model, self).__init__()

# # # # # # #         self.linear1 = nn.Linear(input_dim, 64)
# # # # # # #         self.linear2 = nn.Linear(64, 32)
# # # # # # #         self.linear3 = nn.Linear(32, 16)
# # # # # # #         self.linear4 = nn.Linear(16, output_dim)
# # # # # # #         self.relu = nn.ReLU()
# # # # # # #         self.dropout = nn.Dropout(0.2)
# # # # # # #         self.sig = nn.Sigmoid()

# # # # # # #     def forward(self, x):           #정의 구현 사제단!
# # # # # # #         x = self.linear1(x)
# # # # # # #         x = self.relu(x)
# # # # # # #         x = self.dropout(x)
# # # # # # #         x = self.linear2(x)
# # # # # # #         x = self.relu(x)
# # # # # # #         x = self.linear3(x)
# # # # # # #         x = self.relu(x)
# # # # # # #         x = self.linear4(x)
# # # # # # #         x = self.sig(x)
# # # # # # #         return x
# # # # # # # model = Model(30,1).to(DEVICE)

# # # # # # # criterion= nn.BCELoss()
# # # # # # # optimizer = optim.Adam(model.parameters(), lr = 0.01)

# # # # # # # def train(model, criterion, optimizer, loader):
# # # # # # #     total_loss = 0

# # # # # # #     for x_batch, y_batch in loader:
# # # # # # #         optimizer.zero_grad()
# # # # # # #         hypothesis = model(x_batch)
# # # # # # #         loss = criterion(hypothesis, y_batch)

# # # # # # #         loss.backward()
# # # # # # #         optimizer.step()

# # # # # # #         total_loss += loss.item()

# # # # # # #     return total_loss/len(loader)

# # # # # # # epochs = 300

# # # # # # # for epoch in range(1, epochs+1):
# # # # # # #     loss = train(model, criterion, optimizer, train_loader)
# # # # # # #     print('epochs:{}, loss:{}'.format(epoch, loss))



# # # # # # # def evaluate(model, criterion, loader):
# # # # # # #     model.eval()
# # # # # # #     total_loss = 0

# # # # # # #     for x_batch, y_batch in loader:
# # # # # # #         with torch.no_grad():
# # # # # # #             y_pred = model(x_batch)
# # # # # # #             loss2 = criterion(y_pred, y_batch)

# # # # # # #             total_loss += loss2.item()

# # # # # # #         return total_loss/len(loader)
    
# # # # # # # last_loss = evaluate(model, criterion, test_loader)

# # # # # # # print('최종 loss:', last_loss)

# # # # # # # y_predict = model(x_test)

# # # # # # # y_pred = np.round(y_predict.detach().cpu().numpy())


# # # # # # # y_true = y_test.detach().cpu().numpy()

# # # # # # # accuracy = accuracy_score(y_true, y_pred)
# # # # # # # r2 = r2_score(y_true, y_pred)

# # # # # # # print('accuracy score:', accuracy)


# # # # # # import numpy as np
# # # # # # import torch
# # # # # # import torch.nn as nn
# # # # # # import torch.optim as optim

# # # # # # if torch.cuda.is_available():
# # # # # #     DEVICE = torch.device('cuda')
# # # # # # elif torch.backends.mps.is_available():
# # # # # #     DEVICE = torch.device('mps')
# # # # # # else:
# # # # # #     DEVICE = torch.device('cpu')
    
# # # # # # print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

# # # # # # x_train = np.array([1,2,3,4,5,6,7])
# # # # # # y_train = np.array([1,2,3,4,5,6,7])

# # # # # # x_test = np.array([8,9,10,11])
# # # # # # y_test = np.array([8,9,10,11])
# # # # # # x_pre = np.array([12,13,14])


# # # # # # x_train = torch.tensor(x_train, dtype= torch.float32).unsqueeze(1).to(DEVICE)
# # # # # # x_test = torch.tensor(x_test, dtype= torch.float32).unsqueeze(1).to(DEVICE)
# # # # # # x_pre = torch.tensor(x_pre, dtype= torch.float32).unsqueeze(1).to(DEVICE)
# # # # # # y_train = torch.tensor(y_train, dtype= torch.float32).unsqueeze(1).to(DEVICE)
# # # # # # y_test = torch.tensor(y_test, dtype= torch.float32).unsqueeze(1).to(DEVICE)

# # # # # # x_mean = torch.mean(x_train).to(DEVICE)
# # # # # # x_std = torch.std(x_train).to(DEVICE)

# # # # # # x_train = ((x_train-x_mean)/x_std).to(DEVICE)

# # # # # # model = nn.Sequential(
# # # # # #     nn.Linear(1, 16),
# # # # # #     nn.Linear(16,8),
# # # # # #     nn.Linear(8,1),

# # # # # # ).to(DEVICE)

# # # # # # criterion = nn.MSELoss()
# # # # # # optimizer= optim.Adam(model.parameters(), lr = 0.01)

# # # # # # def train(model, criterion, optimizer, x_train, y_train):
# # # # # #     optimizer.zero_grad()
# # # # # #     hypothesis = model(x_train)
# # # # # #     loss = criterion(hypothesis, y_train)

# # # # # #     loss.backward()
# # # # # #     optimizer.step()

# # # # # #     return loss.item()

# # # # # # epochs = 1000

# # # # # # for epoch in range(1, epochs+1):
# # # # # #     loss = train(model, criterion, optimizer, x_train, y_train)
# # # # # #     print('epoch :{}, loss:{}'.format(epoch, loss))

# # # # # # def evaluate(model, criterion, x_test, y_test):
# # # # # #     model.eval()
# # # # # #     with torch.no_grad():
# # # # # #         y_predict = model(x_test)
# # # # # #         loss2 = criterion(y_test, y_predict)
# # # # # #     return loss2.item()

# # # # # # loss2 = evaluate(model, criterion, x_test, y_test)

# # # # # # x_pred = (x_pre- x_mean)/x_std

# # # # # # result = model(x_pred)

# # # # # # print('xpred 예측값:', result.detach().cpu().numpy())

# # # # # import numpy as np
# # # # # import pandas as pd
# # # # # import torch
# # # # # import torch.nn as nn
# # # # # import torch.optim as optim
# # # # # from sklearn.datasets import load_breast_cancer
# # # # # from sklearn.model_selection import train_test_split
# # # # # from sklearn.preprocessing import StandardScaler
# # # # # from sklearn.metrics import accuracy_score



# # # # # if torch.cuda.is_available():
# # # # #     DEVICE = torch.device('cuda')
# # # # # elif torch.backends.mps.is_available():
# # # # #     DEVICE = torch.device('mps')
# # # # # else:
# # # # #     DEVICE = torch.device('cpu')
    
# # # # # print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)



# # # # # import os

# # # # # if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
# # # # #     BASE_PATH = '/workspace/TensorJae/Study25/'
# # # # # else:                                                 # 로컬인 경우
# # # # #     BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
# # # # # basepath = os.path.join(BASE_PATH)

# # # # # #1 data
# # # # # path = basepath +  '_data/diabetes/'

# # # # # train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# # # # # test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# # # # # sample_submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col= 0)

# # # # # x = train_csv.drop(['Outcome'], axis=1)
# # # # # y = train_csv['Outcome']

# # # # # x = x.replace(0, np.nan)
# # # # # x = x.fillna(train_csv.mean())

# # # # # x_train, x_test, y_train, y_test = train_test_split(
# # # # #     x,y, train_size=0.8, shuffle=True, random_state=337, stratify=y
# # # # # )

# # # # # scaler = StandardScaler()
# # # # # x_train = scaler.fit_transform(x_train)
# # # # # x_test = scaler.transform(x_test)

# # # # # print(x_train.shape)
# # # # # print(y_train.shape)

# # # # # x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
# # # # # x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
# # # # # y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
# # # # # y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)


# # # # # model = nn.Sequential(
# # # # #     nn.Linear(8, 64),
# # # # #     nn.ReLU(),
# # # # #     nn.Linear(64, 32),
# # # # #     nn.ReLU(),
# # # # #     nn.Linear(32, 32),
# # # # #     nn.ReLU(),
# # # # #     nn.Linear(32, 16),
# # # # #     nn.SiLU(),
# # # # #     nn.Linear(16,1),
# # # # #     nn.Sigmoid()
# # # # # ).to(DEVICE)

# # # # # criterion = nn.BCELoss()
# # # # # optimizer = optim.Adam(model.parameters(), lr = 0.01)

# # # # # def train(model, criterion, optimizer, x, y):
# # # # #     optimizer.zero_grad()
# # # # #     hypothesis = model(x)
# # # # #     loss = criterion(hypothesis, y)
# # # # #     loss.backward()
# # # # #     optimizer.step()
    
# # # # #     return loss.item()

# # # # # epochs = 200

# # # # # for epoch in range(1, epochs+1):
# # # # #     loss = train(model, criterion, optimizer, x_train, y_train)
# # # # #     print('epochs:{}, loss:{}'.format(epoch, loss))

# # # # # def evaluate(model, criterion, x,y):
# # # # #     model.eval()

# # # # #     with torch.no_grad():
# # # # #         y_pred = model(x)
# # # # #         loss2 = criterion(y, y_pred)

# # # # #     return loss2.item()

# # # # # last_loss = evaluate(model, criterion, x_test, y_test)
# # # # # print('최종 Loss:', last_loss)

# # # # # y_predict = model(x_test)

# # # # # acc= accuracy_score(y_test.cpu().detach().numpy(), y_predict.cpu().detach().numpy().round())

# # # # # print('accuracy:', acc)



# # # # from sklearn.datasets import fetch_covtype
# # # # from sklearn.model_selection import train_test_split
# # # # from sklearn.preprocessing import StandardScaler
# # # # from sklearn.metrics import accuracy_score
# # # # import torch.nn as nn
# # # # import torch.optim as optim
# # # # import torch
# # # # import pandas as pd
# # # # import numpy as np
# # # # import random
# # # # # # # SEED = 42
# # # # # # # random.seed(SEED)
# # # # # # # np.random.seed(SEED)
# # # # # # # torch.manual_seed(SEED)

# # # # SEED = 42
# # # # random.seed(SEED)
# # # # np.random.seed(SEED)
# # # # torch.manual_seed(SEED)

# # # # if torch.cuda.is_available():
# # # #     DEVICE = torch.device('cuda')
# # # # elif torch.backends.mps.is_available():
# # # #     DEVICE = torch.device('mps')
# # # # else:
# # # #     DEVICE = torch.device('cpu')
    
# # # # print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)



# # # # dataset = fetch_covtype()
# # # # x = dataset.data
# # # # y = dataset.target

# # # # print(dataset.target_names)
# # # # print(np.unique_counts(y))

# # # # print(x.shape, y.shape)


# # # # x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=SEED)

# # # # scaler = StandardScaler()
# # # # x_train = scaler.fit_transform(x_train)
# # # # x_test = scaler.transform(x_test)


# # # # x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
# # # # x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

# # # # y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
# # # # y_test = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

# # # # print(x_train.shape)
# # # # print(y_train.shape)

# # # # #2 model 
# # # # model = nn.Sequential(
# # # #     nn.Linear(54, 64),
# # # #     nn.ReLU(),
# # # #     nn.Linear(64, 32),
# # # #     nn.ReLU(),
# # # #     nn.Linear(32, 32),
# # # #     nn.ReLU(),
# # # #     nn.Linear(32, 16),
# # # #     nn.SiLU(),
# # # #     nn.Linear(16,7),
# # # #     # nn.Softmax() #sparse categorical entropy 를 했기 때문에 마지막 layer linear 로 하면 된다.
# # # # ).to(DEVICE)

# # # # criterion = nn.CrossEntropyLoss()
# # # # optimizer = optim.Adam(model.parameters(), lr = 0.01)

# # # # def train(model, criterion, optimizer, x,y):
# # # #     optimizer.zero_grad()
# # # #     hypothesis = model(x)
# # # #     loss = criterion(hypothesis, y)
# # # #     loss.backward()
# # # #     optimizer.step()

# # # #     return loss.item()

# # # # epochs= 1000

# # # # for epoch in range(1, epochs+1):
# # # #     loss = train(model, criterion, optimizer, x_train, y_train)
# # # #     print('epochs:{}, loss:{}'.format(epoch, loss))


# # # # def evaluate(model, criterion, x,y):
# # # #     model.eval()
# # # #     with torch.no_grad():
# # # #         y_pred = model(x)
# # # #         loss2 = criterion(y_pred, y)
# # # #     return loss2.item()

# # # # last_loss = evaluate(model, criterion, x_test, y_test)

# # # # print('final loss:', last_loss)

# # # # y_predict = model(x_test)
# # # # y_pred = y_predict.detach().cpu().numpy().argmax(axis=1)
# # # # y_pred = y_predict.detach().cpu().numpy().argmax(axis=1)
# # # # y_true = y_test.detach().cpu().numpy()

# # # # acc = accuracy_score(y_true, y_pred)
# # # # print('acc:' ,acc)


# # # from sklearn.datasets import load_digits
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.preprocessing import StandardScaler
# # # from sklearn.metrics import accuracy_score
# # # import torch.nn as nn
# # # import torch.optim as optim

# # # import torch
# # # import random
# # # import numpy as np
# # # import pandas as pd

# # # SEED = 42

# # # random.seed(SEED)
# # # np.random.seed(SEED)


# # # if torch.cuda.is_available():
# # #     DEVICE = torch.device('cuda')
# # # elif torch.backends.mps.is_available():
# # #     DEVICE = torch.device('mps')
# # # else:
# # #     DEVICE = torch.device('cpu')
    
# # # print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)


# # # dataset= load_digits()
# # # x = dataset.data
# # # y = dataset.target

# # # print(np.unique_counts(y))

# # # x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=SEED)



# # # scaler = StandardScaler()
# # # x_train = scaler.fit_transform(x_train)
# # # x_test = scaler.transform(x_test)

# # # x_train = torch.tensor(x_train, dtype= torch.float32).to(DEVICE)
# # # x_test = torch.tensor(x_test, dtype= torch.float32).to(DEVICE)
# # # y_train = torch.tensor(y_train, dtype= torch.long).to(DEVICE)
# # # y_test = torch.tensor(y_test, dtype= torch.long).to(DEVICE)
# # # print(x_train.shape)
# # # print(y_train.shape)



# # # class Model(nn.Module):
# # #     def __init__(self, input_dim, output_dim):
# # #         super(Model, self).__init__()
        
# # #         self.linear1 = nn.Linear(input_dim, 64) 
# # #         self.linear2 = nn.Linear(64, 32)
# # #         self.linear3 = nn.Linear(32, 32)
# # #         self.linear4 = nn.Linear(32,16)
# # #         self.linear5 = nn.Linear(16, output_dim)
# # #         self.relu = nn.ReLU()
# # #         self.silu = nn.SiLU()
# # #         self.dropout = nn.Dropout(0.2)
        
# # #     def forward(self, x):
# # #         x = self.linear1(x)
# # #         x = self.relu(x)
# # #         x = self.linear2(x)
# # #         x = self.relu(x)
# # #         x = self.linear3(x)
# # #         x = self.relu(x)
# # #         x = self.linear3(x)
# # #         x = self.relu(x)
# # #         x = self.linear3(x)
# # #         x = self.relu(x)
# # #         x = self.linear4(x)
# # #         x = self.silu(x)
# # #         x = self.linear5(x)
# # #         return x
    

# # # model = Model(64,10).to(DEVICE) 


# # # criterion= nn.CrossEntropyLoss()
# # # optimizer = optim.Adam(model.parameters(), lr =0.01)

# # # def train(model, criterion, optimizer, x,y):
# # #     optimizer.zero_grad()
# # #     hypothesis = model(x)
# # #     loss = criterion(hypothesis, y)
# # #     loss.backward()
# # #     optimizer.step()
# # #     return loss.item()

# # # epochs = 100

# # # for epoch in range(1, epochs+1):
# # #     loss = train(model, criterion, optimizer, x_train, y_train)
# # #     print('epoch:{}, loss:{}'.format(epoch, loss))

# # # def evaluate(model, criterion, x,y):
# # #     model.eval()

# # #     with torch.no_grad():
# # #         y_pred = model(x)
# # #         loss2 = criterion(y_pred, y)

# # #     return loss2.item()

# # # last_loss = evaluate(model, criterion, x_test, y_test)

# # # print('last loss:', last_loss)


# # # y_predict = model(x_test)
# # # y_pred = y_predict.detach().cpu().numpy().argmax(axis=1)
# # # y_true = y_test.detach().cpu().numpy()

# # # acc = accuracy_score(y_true, y_pred)

# # # print('acc:', acc)


# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from sklearn.datasets import load_breast_cancer
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.metrics import accuracy_score
# # from sklearn.metrics import r2_score
# # import random

# # SEED = 42
# # random.seed(SEED)
# # np.random.seed(SEED)
# # torch.manual_seed(SEED)


# # if torch.cuda.is_available():
# #     DEVICE = torch.device('cuda')
# # elif torch.backends.mps.is_available():
# #     DEVICE = torch.device('mps')
# # else:
# #     DEVICE = torch.device('cpu')
    
# # print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

# # # 1. 데이터
# # datasets = np.array([1,2,3,4,5,6,7,8,9,10])
 
# # x = np.array([[1,2,3],
# #              [2,3,4],
# #              [3,4,5],
# #              [4,5,6],
# #              [5,6,7],
# #              [6,7,8],
# #              [7,8,9],])        # (7, 3)
# # y = np.array([4,5,6,7,8,9,10])

# # print(x.shape, y.shape)



# # x = x.reshape(x.shape[0], x.shape[1], 1)
# # x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
# # y = torch.tensor(y, dtype=torch.float32).to(DEVICE)

# # from torch.utils.data import TensorDataset, DataLoader
# # train_set = TensorDataset(x,y)
# # train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

# # aaa = iter(train_loader)
# # bbb = next(aaa)


# # exit()

# # class RNN(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.rnn_layer1 = nn.RNN(
# #             input_size=1,
# #             hidden_size=32,
# #             num_layers=1,
# #             batch_first=True
# #         )



# import os
# import random
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from rdkit import Chem
# from rdkit.Chem import Descriptors, AllChem, DataStructs
# from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# from torch_geometric.nn import NNConv, global_mean_pool
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor
# from catboost import CatBoostRegressor

# # 1) 시드 고정
# seed = 9792
# def seed_everything(s):
#     random.seed(s); np.random.seed(s)
#     os.environ['PYTHONHASHSEED'] = str(s)
#     torch.manual_seed(s); torch.cuda.manual_seed_all(s)
#     torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
# seed_everything(seed)

# # 2) 데이터 로드
# path = '/Users/jaewoo000/Desktop/IBM:RedHat/Study25/_data/dacon/drugs/'
# train = pd.read_csv(os.path.join(path, "train.csv"))
# test  = pd.read_csv(os.path.join(path, "test.csv"))
# y     = train['Inhibition'].values

# # 3) Scaffold split
# def scaffold_split(df, smiles_col='Canonical_Smiles', n_folds=5):
#     scaffolds = {}
#     for i, smi in enumerate(df[smiles_col]):
#         scaf = MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
#         scaffolds.setdefault(scaf, []).append(i)
#     groups = sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)
#     folds = [[] for _ in range(n_folds)]
#     for idx, (_, idxs) in enumerate(groups):
#         folds[idx % n_folds].extend(idxs)
#     return folds

# folds = scaffold_split(train, n_folds=5)

# # 4) SMILES 증강
# def smiles_augment(smiles, n_aug=5):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None: return [smiles]*n_aug
#     out = {Chem.MolToSmiles(mol, canonical=True)}
#     while len(out) < n_aug:
#         out.add(Chem.MolToSmiles(mol, doRandom=True))
#     return list(out)

# # 5) 전역 피처
# def global_descriptors(smi):
#     mol = Chem.MolFromSmiles(smi)
#     if mol is None:
#         return np.zeros(10, dtype=np.float32)
#     vals = [
#         Descriptors.MolWt(mol),
#         Descriptors.MolLogP(mol),
#         Descriptors.TPSA(mol),
#         Descriptors.NumRotatableBonds(mol),
#         Descriptors.NumHAcceptors(mol),
#         Descriptors.NumHDonors(mol),
#         Descriptors.HeavyAtomCount(mol),
#         Descriptors.RingCount(mol),
#         Descriptors.FractionCSP3(mol),
#         sum(int(a.GetIsAromatic()) for a in mol.GetAtoms())/max(1,mol.GetNumAtoms())
#     ]
#     return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

# # 6) Morgan fingerprint (2048bit)
# def morgan_fp(smi, radius=2, nBits=2048):
#     arr = np.zeros((nBits,), dtype=np.uint8)
#     mol = Chem.MolFromSmiles(smi)
#     if mol is None: return arr
#     fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
#     DataStructs.ConvertToNumpyArray(fp, arr)
#     return arr

# # 7) SMILES→Graph
# def mol_to_graph(smi, lbl=None):
#     mol = Chem.MolFromSmiles(smi)
#     if mol is None: return None
#     nf, ei, ea = [], [], []
#     for atom in mol.GetAtoms():
#         nf.append([
#             atom.GetAtomicNum(),
#             atom.GetTotalDegree(),
#             atom.GetFormalCharge(),
#             int(atom.GetHybridization()),
#             int(atom.GetIsAromatic())
#         ])
#     for bond in mol.GetBonds():
#         i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#         bt  = bond.GetBondTypeAsDouble()
#         ei += [[i,j],[j,i]]
#         ea += [[bt],[bt]]
#     if not ei: return None
#     return Data(
#         x=torch.tensor(nf, dtype=torch.float32),
#         edge_index=torch.tensor(ei,dtype=torch.long).t().contiguous(),
#         edge_attr=torch.tensor(ea,dtype=torch.float32),
#         global_feat=torch.tensor(global_descriptors(smi),dtype=torch.float32).view(1,-1),
#         y=None if lbl is None else torch.tensor([lbl],dtype=torch.float32)
#     )

# # 8) GNN 정의 (3-layer NNConv)
# class MPNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.ec1 = nn.Sequential(nn.Linear(1,128), nn.SiLU(), nn.Linear(128,5*128))
#         self.conv1 = NNConv(5,128,self.ec1,aggr='mean')
#         self.ec2 = nn.Sequential(nn.Linear(1,128), nn.SiLU(), nn.Linear(128,128*128))
#         self.conv2 = NNConv(128,128,self.ec2,aggr='mean')
#         self.ec3 = nn.Sequential(nn.Linear(1,128), nn.SiLU(), nn.Linear(128,128*128))
#         self.conv3 = NNConv(128,128,self.ec3,aggr='mean')
#         self.gp   = nn.Sequential(nn.Linear(10,32), nn.BatchNorm1d(32), nn.SiLU())
#         self.fc   = nn.Sequential(
#             nn.Linear(128+32,128), nn.BatchNorm1d(128), nn.SiLU(),
#             nn.Dropout(0.2), nn.Linear(128,1)
#         )
#     def forward(self, d):
#         x = F.silu(self.conv1(d.x,d.edge_index,d.edge_attr))
#         x = F.silu(self.conv2(x,d.edge_index,d.edge_attr))
#         x = F.silu(self.conv3(x,d.edge_index,d.edge_attr))
#         x = global_mean_pool(x, d.batch)
#         g = self.gp(d.global_feat.to(x.device)).expand(x.size(0),-1)
#         return self.fc(torch.cat([x,g],1)).squeeze()

# # 9) Loss & routines
# def weighted_mse(p,t):
#     w = torch.ones_like(t)
#     w[t<5]  = 2.5
#     w[t>95] = 2.5
#     return (w*(p-t)**2).mean()

# def train_epoch(m,ldr,opt,dev):
#     m.train(); tot,cnt=0.0,0
#     for b in ldr:
#         b=b.to(dev); opt.zero_grad()
#         loss=weighted_mse(m(b),b.y.view(-1))
#         loss.backward(); opt.step()
#         tot+=loss.item()*b.num_graphs; cnt+=b.num_graphs
#     return tot/cnt

# def eval_epoch(m,ldr,dev):
#     m.eval(); ps,ts=[],[]
#     with torch.no_grad():
#         for b in ldr:
#             b=b.to(dev)
#             ps.append(m(b).cpu().numpy())
#             ts.append(b.y.view(-1).cpu().numpy())
#     return np.concatenate(ps), np.concatenate(ts)

# # 10) Classical features
# X_desc = np.vstack([global_descriptors(s) for s in train['Canonical_Smiles']])
# X_fp   = np.vstack([morgan_fp(s) for s in train['Canonical_Smiles']])
# X_all  = np.hstack([X_desc, X_fp])

# # 11) OOF buffers
# N = len(train)
# oof_gnn   = np.zeros(N, dtype=np.float32)
# oof_ridge = np.zeros(N, dtype=np.float32)
# oof_rf    = np.zeros(N, dtype=np.float32)
# oof_cb    = np.zeros(N, dtype=np.float32)
# fold_states = []

# # classical params
# cb_params = dict(
#     iterations=1500, depth=8, learning_rate=0.04,
#     l2_leaf_reg=5, subsample=0.8,
#     random_seed=seed, verbose=0,
#     early_stopping_rounds=50
# )

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 12) Scaffold OOF loop
# for fold_idx in range(5):
#     tr_idx = [i for f in range(5) if f!=fold_idx for i in folds[f]]
#     va_idx = folds[fold_idx]

#     # GNN
#     tr_graphs = []
#     for i in tr_idx:
#         smi,lab = train.loc[i,'Canonical_Smiles'], y[i]
#         for aug in smiles_augment(smi,5):
#             g=mol_to_graph(aug,lab)
#             if g: tr_graphs.append(g)
#     va_graphs = [mol_to_graph(train.loc[i,'Canonical_Smiles'],y[i]) for i in va_idx]
#     va_graphs = [g for g in va_graphs if g]

#     tr_ld = DataLoader(tr_graphs,batch_size=64,shuffle=True)
#     va_ld = DataLoader(va_graphs,batch_size=128)

#     model = MPNN().to(device)
#     opt   = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
#     sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,'min',patience=5,factor=0.5)
#     best_rmse,best_st = np.inf, None
#     for _ in range(60):
#         train_epoch(model,tr_ld,opt,device)
#         p,t = eval_epoch(model,va_ld,device)
#         rmse = np.sqrt(mean_squared_error(t,p))
#         sch.step(rmse)
#         if rmse<best_rmse:
#             best_rmse,best_st = rmse,model.state_dict()
#     fold_states.append(best_st)
#     model.load_state_dict(best_st)
#     p,_ = eval_epoch(model,va_ld,device)
#     oof_gnn[va_idx] = p[:len(va_idx)]

#     # Classical
#     X_tr, X_va = X_all[tr_idx], X_all[va_idx]
#     y_tr, y_va = y[tr_idx], y[va_idx]

#     sc = StandardScaler().fit(X_tr)
#     oof_ridge[va_idx] = Ridge(alpha=5.0,random_state=seed)\
#         .fit(sc.transform(X_tr),y_tr)\
#         .predict(sc.transform(X_va))
#     oof_rf[va_idx] = RandomForestRegressor(
#         n_estimators=300,max_depth=12,min_samples_leaf=3,
#         random_state=seed,n_jobs=1
#     ).fit(X_tr,y_tr).predict(X_va)
#     cb = CatBoostRegressor(**cb_params)
#     cb.fit(X_tr,y_tr,eval_set=(X_va,y_va),verbose=False)
#     oof_cb[va_idx] = cb.predict(X_va)

# # 13) Meta-model (CatBoost)
# meta_X = np.vstack([oof_gnn,oof_ridge,oof_rf,oof_cb]).T
# meta = CatBoostRegressor(
#     iterations=1500, depth=6, learning_rate=0.02,
#     random_seed=seed, verbose=0, early_stopping_rounds=50
# )
# meta.fit(meta_X, y)

# # 14) 검증 지표
# val_preds = meta.predict(meta_X)
# val_rmse  = np.sqrt(mean_squared_error(y,val_preds))
# val_r2    = r2_score(y,val_preds)
# val_score = 0.5*(1-min(val_rmse/100,1))+0.5*val_r2
# print(f"✅ Validation RMSE:   {val_rmse:.4f}")
# print(f"✅ Validation R2:     {val_r2:.4f}")
# print(f"✅ Validation Score:  {val_score:.4f}")

# # 15) Retrain classical on full data
# sc_full = StandardScaler().fit(X_all)
# ridge_f = Ridge(alpha=5.0,random_state=seed).fit(sc_full.transform(X_all),y)
# rf_f    = RandomForestRegressor(
#               n_estimators=300,max_depth=12,min_samples_leaf=3,
#               random_state=seed,n_jobs=1
#           ).fit(X_all,y)
# cb_f    = CatBoostRegressor(**cb_params).fit(X_all,y)

# # 16) Test 예측
# test_gnn = []
# for smi in tqdm(test['Canonical_Smiles'],desc="GNN TTA"):
#     preds=[]
#     for st in fold_states:
#         m = MPNN().to(device)
#         m.load_state_dict(st); m.eval()
#         tta=[]
#         for aug in smiles_augment(smi,20):  # ← TTA 20회
#             g=mol_to_graph(aug)
#             if not g: continue
#             g=g.to(device)
#             g.batch=torch.zeros(g.x.size(0),dtype=torch.long,device=device)
#             with torch.no_grad():
#                 tta.append(m(g).cpu().item())
#         if tta:
#             preds.append(np.mean(tta))
#     test_gnn.append(np.mean(preds) if preds else 0.0)

# # Classical test
# X_desc_test = np.vstack([global_descriptors(s) for s in test['Canonical_Smiles']])
# X_fp_test   = np.vstack([morgan_fp(s) for s in test['Canonical_Smiles']])
# X_test_all  = np.hstack([X_desc_test, X_fp_test])

# ridge_pred = ridge_f.predict(sc_full.transform(X_test_all))
# rf_pred    = rf_f.predict(X_test_all)
# cb_pred    = cb_f.predict(X_test_all)

# # Meta test & submission
# meta_test   = np.vstack([test_gnn,ridge_pred,rf_pred,cb_pred]).T
# final_preds = np.clip(meta.predict(meta_test),0,100)

# pd.DataFrame({
#     "ID":         test['ID'],
#     "Inhibition": final_preds
# }).to_csv(os.path.join(path,"submission_scaffold_catmeta_v2.csv"),index=False)
# print("✅ submission_scaffold_catmeta_v2.csv 저장 완료")


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
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from optuna.samplers import TPESampler
import json
warnings.filterwarnings(action='ignore')

# Seed 고정
seed = 6054
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 경로 설정
BASE_PATH = '/workspace/TensorJae/Study25/' if os.path.exists('/workspace/TensorJae/Study25/') else os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
path = os.path.join(BASE_PATH, '_data/dacon/electricity/')
buildinginfo = pd.read_csv(path + 'building_info.csv')
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
samplesub = pd.read_csv(path + 'sample_submission.csv')

# 결측치 처리
for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    buildinginfo[col] = buildinginfo[col].replace('-', 0).astype(float)
    
buildinginfo['solar_power_utility'] = np.where(buildinginfo['태양광용량(kW)'] != 0, 1, 0)
buildinginfo['ess_utility'] = np.where(buildinginfo['ESS저장용량(kWh)'] != 0, 1, 0)
buildinginfo['pcs_utility'] = np.where(buildinginfo['PCS용량(kW)'] != 0, 1, 0)

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

# 💡 [개선 전략] '일사량' 대리 변수 생성
# 1. 훈련 데이터에서 월(month)과 시간(hour)별 평균 일사량 계산
solar_proxy = train.groupby(['month', 'hour'])['일사(MJ/m2)'].mean().reset_index()
solar_proxy.rename(columns={'일사(MJ/m2)': 'expected_solar'}, inplace=True)

# 2. train과 test 데이터에 'expected_solar' 피처를 병합
train = train.merge(solar_proxy, on=['month', 'hour'], how='left')
test = test.merge(solar_proxy, on=['month', 'hour'], how='left')

# 병합 과정에서 발생할 수 있는 소량의 결측치는 0으로 채웁니다.
train['expected_solar'] = train['expected_solar'].fillna(0)
test['expected_solar'] = test['expected_solar'].fillna(0)

train['건물유형'] = train['건물유형'].astype('category').cat.codes
test['건물유형'] = test['건물유형'].astype('category').cat.codes

features = [
    '건물유형', '연면적(m2)', '냉방면적(m2)', 
    # '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
    'solar_power_utility', 'ess_utility', 'pcs_utility',
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
    'hour', 'dayofweek', 'month', 'day', 'is_weekend',
    'is_working_hours', 'sin_hour', 'cos_hour', 'DI', 'expected_solar'
]

target = '전력소비량(kWh)'

# Optuna 튜닝 함수들
def tune_xgb(trial, x_train, y_train, x_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'early_stopping_rounds': 50,
        'eval_metric': 'mae',
        'random_state': seed,
        'objective': 'reg:squarederror'
    }
    model = XGBRegressor(**params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    pred = model.predict(x_val)
    smape = np.mean(200 * np.abs(np.expm1(pred) - np.expm1(y_val)) /
                    (np.abs(np.expm1(pred)) + np.abs(np.expm1(y_val)) + 1e-6))
    return smape

def tune_lgb(trial, x_train, y_train, x_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': seed,
        'objective': 'mae'
    }
    model = LGBMRegressor(**params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = model.predict(x_val)
    smape = np.mean(200 * np.abs(np.expm1(pred) - np.expm1(y_val)) /
                    (np.abs(np.expm1(pred)) + np.abs(np.expm1(y_val)) + 1e-6))
    return smape

def tune_cat(trial, x_train, y_train, x_val, y_val):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'random_seed': seed,
        'loss_function': 'MAE',
        'verbose': 0
    }
    model = CatBoostRegressor(**params)
    model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=50, verbose=0)
    pred = model.predict(x_val)
    smape = np.mean(200 * np.abs(np.expm1(pred) - np.expm1(y_val)) /
                    (np.abs(np.expm1(pred)) + np.abs(np.expm1(y_val)) + 1e-6))
    return smape

def objective(trial, oof_train, oof_val, y_train, y_val):
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    ridge = Ridge(alpha=alpha)
    ridge.fit(oof_train, y_train)
    preds = ridge.predict(oof_val)
    smape = np.mean(200 * np.abs(np.expm1(preds) - np.expm1(y_val)) /
                    (np.abs(np.expm1(preds)) + np.abs(np.expm1(y_val)) + 1e-6))
    return smape

def process_building_kfold(bno):
    print(f"🏢 건물번호 {bno} KFold 처리 중...")
    param_dir = os.path.join(path, "optuna_params")  # ✅ 추가
    os.makedirs(param_dir, exist_ok=True)   
    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()
    x = train_b[features].values
    target_values = np.array(train_b[target])
    y = np.log1p(target_values)
    x_test = test_b[features].values

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    test_preds = []
    val_smapes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
        print(f" - Fold {fold+1}")
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test)

        # 기존 코드 지우고 아래로 교체
        model_key = f"{bno}_fold{fold+1}_xgb"
        xgb_param_path = os.path.join(param_dir, f"{model_key}.json")

        if os.path.exists(xgb_param_path):
            with open(xgb_param_path, "r") as f:
                xgb_params = json.load(f)
        else:
            xgb_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
            xgb_study.optimize(lambda trial: tune_xgb(trial, x_train_scaled, y_train, x_val_scaled, y_val), n_trials=20)
            xgb_params = xgb_study.best_params
            with open(xgb_param_path, "w") as f:
                json.dump(xgb_params, f)

        best_xgb = XGBRegressor(**xgb_params)
        best_xgb.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

        model_key = f"{bno}_fold{fold+1}_lgb"
        lgb_param_path = os.path.join(param_dir, f"{model_key}.json")

        if os.path.exists(lgb_param_path):
            with open(lgb_param_path, "r") as f:
                lgb_params = json.load(f)
        else:
            lgb_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
            lgb_study.optimize(lambda trial: tune_lgb(trial, x_train_scaled, y_train, x_val_scaled, y_val), n_trials=20)
            lgb_params = lgb_study.best_params
            with open(lgb_param_path, "w") as f:
                json.dump(lgb_params, f)

        best_lgb = LGBMRegressor(**lgb_params)
        best_lgb.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])

        model_key = f"{bno}_fold{fold+1}_cat"
        cat_param_path = os.path.join(param_dir, f"{model_key}.json")

        if os.path.exists(cat_param_path):
            with open(cat_param_path, "r") as f:
                cat_params = json.load(f)
        else:
            cat_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
            cat_study.optimize(lambda trial: tune_cat(trial, x_train_scaled, y_train, x_val_scaled, y_val), n_trials=20)
            cat_params = cat_study.best_params
            with open(cat_param_path, "w") as f:
                json.dump(cat_params, f)

        best_cat = CatBoostRegressor(**cat_params)
        best_cat.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50, verbose=0)

        # Stacking용 예측값 생성
        oof_train = np.vstack([
            best_xgb.predict(x_train_scaled),
            best_lgb.predict(x_train_scaled),
            best_cat.predict(x_train_scaled)
        ]).T
        oof_val = np.vstack([
            best_xgb.predict(x_val_scaled),
            best_lgb.predict(x_val_scaled),
            best_cat.predict(x_val_scaled)
        ]).T
        oof_test = np.vstack([
            best_xgb.predict(x_test_scaled),
            best_lgb.predict(x_test_scaled),
            best_cat.predict(x_test_scaled)
        ]).T

        # Ridge 튜닝 및 학습
        model_key = f"{bno}_fold{fold+1}_ridge"
        ridge_param_path = os.path.join(param_dir, f"{model_key}.json")

        if os.path.exists(ridge_param_path):
            with open(ridge_param_path, "r") as f:
                ridge_params = json.load(f)
        else:
            ridge_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
            ridge_study.optimize(lambda trial: objective(trial, oof_train, oof_val, y_train, y_val), n_trials=30)
            ridge_params = ridge_study.best_params
            with open(ridge_param_path, "w") as f:
                json.dump(ridge_params, f)

        meta = Ridge(alpha=ridge_params['alpha'])
        meta.fit(oof_train, y_train)

        val_pred = meta.predict(oof_val)
        test_pred = meta.predict(oof_test)

        smape = np.mean(200 * np.abs(np.expm1(val_pred) - np.expm1(y_val)) /
                        (np.abs(np.expm1(val_pred)) + np.abs(np.expm1(y_val)) + 1e-6))

        val_smapes.append(smape)
        test_preds.append(np.expm1(test_pred))

    # 평균 예측값과 SMAPE
    avg_test_pred = np.mean(test_preds, axis=0)
    avg_smape = np.mean(val_smapes)

    return avg_test_pred.tolist(), avg_smape

# 병렬 처리 실행 (KFold 적용)
results = Parallel(n_jobs=-10, backend='loky')(
    delayed(process_building_kfold)(bno) for bno in train['건물번호'].unique()
)

# 결과 합치기
final_preds = []
val_smapes = []
for preds, smape in results:
    final_preds.extend(preds)
    val_smapes.append(smape)

samplesub['answer'] = final_preds
today = datetime.datetime.now().strftime('%Y%m%d')
avg_smape = np.mean(val_smapes)
filename = f"submission_stack_optuna_{today}_SMAPE_일사_{avg_smape:.4f}_{seed}.csv"
samplesub.to_csv(os.path.join(path, filename), index=False)

print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
print(f"📁 저장 완료 → {filename}")