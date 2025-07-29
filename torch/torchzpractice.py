# # # # # # # # # import torch
# # # # # # # # # import torch.version


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

# # # # # # # # # x = np.array([1,2,3])
# # # # # # # # # y = np.array([1,2,3])

# # # # # # # # # x = torch.FloatTensor(x)
# # # # # # # # # x = x.reshape(-1,1)
# # # # # # # # # y = torch.FloatTensor(y).unsqueeze(1)

# # # # # # # # # model = nn.Linear(1,1) #input output

# # # # # # # # # criterion = nn.MSELoss()
# # # # # # # # # optimizer = optim.SGD(model.parameters(), lr = 0.1)

# # # # # # # # # def train(model, criterion, optimizer, x, y):
# # # # # # # # #     optimizer.zero_grad()#기울기 초기화 각 배치마다 기울기를 초기화하여, 기울기 누적에 의한 문제 해결
# # # # # # # # #     hypothesis = model(x)   #y = xw+b
# # # # # # # # #     loss = criterion(hypothesis, y) # loss = mse() = 시그마 (y-hypothesis)^2/n
# # # # # # # # #     # 	•	모델에 x를 넣어서 예측값을 얻습니다.
# # # # # # # # # 	# •	이 예측값을 가설(hypothesis) 이라고 부릅니다.
    
# # # # # # # # #     loss.backward() #기울기 값까지만 계산
# # # # # # # # #     optimizer.step()    # 가중치 갱신
# # # # # # # # #     return loss.item()

# # # # # # # # # epochs = 700
# # # # # # # # # for epoch in range(1, epochs+1):
# # # # # # # # #     loss = train(model, criterion, optimizer, x, y)
# # # # # # # # #     print('epoch: {}, loss: {}'.format(epoch, loss))    

# # # # # # # # # def evaluate(model, criterion, x, y):
# # # # # # # # #     model.eval()
# # # # # # # # #     with torch.no_grad():
# # # # # # # # #         y_predict = model(x)
# # # # # # # # #         loss2 = criterion(y, y_predict)
# # # # # # # # #     return loss2.item()

# # # # # # # # # loss2 = evaluate(model, criterion, x, y)
# # # # # # # # # print('최종 loss:', loss2)

# # # # # # # # # result = model(torch.Tensor([[4]]))
# # # # # # # # # print('4의 예측값:', result)
# # # # # # # # # print('4의 예측값', result.item())


# # # # # # # # import numpy as np
# # # # # # # # import torch
# # # # # # # # import torch.nn as nn
# # # # # # # # import torch.optim as optim

# # # # # # # # x = np.array([1,2,3])
# # # # # # # # y = np.array([1,2,3])

# # # # # # # # x = torch.FloatTensor(x)


# # # # # # # # x = torch.FloatTensor(x).unsqueeze(1)

# # # # # # # # y = torch.FloatTensor(y).unsqueeze(1)

# # # # # # # # model = nn.Linear(1,1)

# # # # # # # # criterion = nn.MSELoss()

# # # # # # # # optimizer = optim.SGD(model.parameters(), lr=0.1)

# # # # # # # # def train(model, criterion, optimizer, x, y):
# # # # # # # #     optimizer.zero_grad()
# # # # # # # #     hypothesis = model(x)
# # # # # # # #     loss = criterion(hypothesis, y)
# # # # # # # #     loss.backward()
# # # # # # # #     optimizer.step()
# # # # # # # #     return loss.item()


# # # # # # # # epochs = 1000,
# # # # # # # # for epoch in range(1,epochs+1):
# # # # # # # #     loss = train(model, criterion, optimizer, x,y,)
# # # # # # # #     print('epoch:{}, loss:{}'.format(epoch, loss))
    
# # # # # # # # def evaluate(model, criterion, x, y):
# # # # # # # #     model.eval()
# # # # # # # #     with torch.no_grad():
# # # # # # # #         y_predict = model(x)
# # # # # # # #         loss2 = criterion(y, y_predict)
        
# # # # # # # #     return loss2.item()


# # # # # # # # loss2 = evaluate(model, criterion, x,y)
# # # # # # # # print('최종 loss:', loss2)

# # # # # # # # result = model(torch.Tensor([[4]]))
# # # # # # # # print('4의 예측값:', result)
# # # # # # # # print('')

# # # # # # # import numpy as np
# # # # # # # import torch
# # # # # # # import torch.nn as nn
# # # # # # # import torch.optim as optim

# # # # # # # if torch.cuda.is_available():
# # # # # # #     DEVICE = torch.device('cuda')
# # # # # # # elif torch.backends.mps.is_available():
# # # # # # #     DEVICE = torch.device('mps')
# # # # # # # else:
# # # # # # #     DEVICE = torch.device('cpu')
    
# # # # # # # print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

# # # # # # # x = np.array([[1,2,3,4,5,6,7,8,9,10],[1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]]).transpose()
# # # # # # # y = np.array([1,2,3,4,5,6,7,7,9,10])


# # # # # # # x = torch.FloatTensor(x).to(DEVICE)

# # # # # # # y = y.reshape(-1,1)
# # # # # # # y = torch.FloatTensor(y).to(DEVICE)

# # # # # # # x_mean = torch.mean(x)
# # # # # # # x_std = torch.std(x)

# # # # # # # x = (x-torch.mean(x))/ torch.std(x)

# # # # # # # model = nn.Sequential(
# # # # # # #     nn.Linear(2, 5),
# # # # # # #     nn.Linear(5,4),
# # # # # # #     nn.Linear(4,3),
# # # # # # #     nn.Linear(3,2),
# # # # # # #     nn.Linear(2,1),
    
# # # # # # # ).to(DEVICE)


# # # # # # # criterion= nn.MSELoss()

# # # # # # # optimizer = optim.SGD(model.parameters(), lr= 0.01)

# # # # # # # def train(model, criterion, optimizer, x,y):
# # # # # # #     optimizer.zero_grad()
# # # # # # #     hypothesis = model(x)
# # # # # # #     loss = criterion(hypothesis, y)
# # # # # # #     loss.backward()
# # # # # # #     optimizer.step()
# # # # # # #     return loss.item()

# # # # # # # epochs = 1000
# # # # # # # for epoch in range(1, epochs+1):
# # # # # # #     loss = train(model, criterion,optimizer, x,y)
# # # # # # #     print('epoch:{}, loss:{}'.format(epoch, loss))

# # # # # # # def evaluate(model, criterion, x,y):
# # # # # # #     model.eval()
# # # # # # #     with torch.no_grad():
# # # # # # #         y_predict = model(x)
# # # # # # #         loss2 = criterion(y, y_predict)
# # # # # # #     return loss2.item()

# # # # # # # loss2 = evaluate(model, criterion, x,y) 
# # # # # # # print('최종 loss:', loss2)

# # # # # # # x_pred = (torch.Tensor([[10,1.3]]).to(DEVICE)- x_mean)/x_std
# # # # # # # result = model(x_pred)

# # # # # # # print('10, 1.3의 예측값:', result)
# # # # # # # print('10, 1.3의 예측값', result.item())   


# # # # # # import numpy as np
# # # # # # import torch
# # # # # # import torch.nn as nn
# # # # # # import torch.optim as optim
# # # # # # from sklearn.datasets import load_breast_cancer, fetch_california_housing
# # # # # # from sklearn.model_selection import train_test_split
# # # # # # from sklearn.preprocessing import StandardScaler
# # # # # # from sklearn.metrics import accuracy_score
# # # # # # from sklearn.metrics import r2_score
# # # # # # import random

# # # # # # SEED = 42
# # # # # # import random
# # # # # # random.seed(SEED)
# # # # # # np.random.seed(SEED)
# # # # # # torch.manual_seed(SEED)
# # # # # # torch.cuda.manual_seed(SEED)


# # # # # # if torch.cuda.is_available():
# # # # # #     DEVICE= torch.device('cuda')
# # # # # # elif torch.backends.mps.is_available():
# # # # # #     DEVICE = torch.device('mps')
# # # # # # else:
# # # # # #     DEVICE = torch.device('cpu')


# # # # # # print('torch:', torch.__version__, '사용 device:', DEVICE)

# # # # # # dataset = fetch_california_housing()
# # # # # # x = dataset.data
# # # # # # y = dataset.target

# # # # # # x_train, x_test, y_train, y_test  = train_test_split(
# # # # # #     x,y, train_size=0.8, shuffle=True, random_state=SEED
# # # # # # )

# # # # # # scaler = StandardScaler()
# # # # # # x_train = scaler.fit_transform(x_train)
# # # # # # x_test = scaler.transform(x_test)

# # # # # # x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
# # # # # # x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
# # # # # # y_train = torch.tensor(y_train, dtype= torch.float32).unsqueeze(1).to(DEVICE)
# # # # # # y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

# # # # # # print(x_train.dtype)
# # # # # # print(x_train.shape, y_train.shape)
# # # # # # print(type())

# # # # # import numpy as np
# # # # # import torch
# # # # # import torch.nn as nn
# # # # # import torch.optim as optim
# # # # # from sklearn.datasets import load_breast_cancer
# # # # # from sklearn.preprocessing import StandardScaler
# # # # # from sklearn.model_selection import train_test_split
# # # # # from sklearn.metrics import accuracy_score
# # # # # from sklearn.metrics import r2_score
# # # # # import random
# # # # # SEED = 42
# # # # # random.seed(SEED)
# # # # # np.random.seed(SEED)
# # # # # torch.manual_seed(SEED)

# # # # # if torch.cuda.is_available():
# # # # #     DEVICE = torch.device('cuda')
# # # # # elif torch.backends.mps.is_available():
# # # # #     DEVICE = torch.device('mps')
# # # # # else:
# # # # #     DEVICE = torch.device('cpu')
    
# # # # # print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)


# # # # # dataset = load_breast_cancer()
# # # # # x = dataset.data
# # # # # y = dataset.target

# # # # # print(x.shape)
# # # # # print(y.shape)

# # # # # x_train, x_test, y_train, y_test = train_test_split(
# # # # #     x, y, train_size=0.8, shuffle=True, random_state=SEED
# # # # # )

# # # # # scaler = StandardScaler()
# # # # # x_train = scaler.fit_transform(x_train)
# # # # # x_test = scaler.transform(x_test)

# # # # # x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
# # # # # x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
# # # # # y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
# # # # # y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)


# # # # # print(x_train.dtype)
# # # # # print(x_train.shape, y_train.shape)
# # # # # print(type(x_train))

# # # # # from torch.utils.data import TensorDataset
# # # # # from torch.utils.data import DataLoader

# # # # # train_set = TensorDataset(x_train, y_train)
# # # # # test_set = TensorDataset(x_test, y_test)

# # # # # print(train_set)
# # # # # print(type(train_set))
# # # # # print(len(train_set))
# # # # # print(train_set[0])

# # # # # print(train_set[0][0])
# # # # # print(train_set[0][1])

# # # # # train_loader = DataLoader(train_set, batch_size=100, shuffle=False)
# # # # # test_loader = DataLoader(test_set, batch_size=100, shuffle=False)


# # # # # class Model(nn.Module):
# # # # #     def __init__(self, input_dim, output_dim):
# # # # #         super(Model, self).__init__()

# # # # #         self.linear1 = nn.Linear(input_dim, 64)
# # # # #         self.linear2 = nn.Linear(64, 32)
# # # # #         self.linear3 = nn.Linear(32, 16)
# # # # #         self.linear4 = nn.Linear(16, output_dim)
# # # # #         self.relu = nn.ReLU()
# # # # #         self.dropout = nn.Dropout(0.2)
# # # # #         self.sig = nn.Sigmoid()

# # # # #     def forward(self, x):           #정의 구현 사제단!
# # # # #         x = self.linear1(x)
# # # # #         x = self.relu(x)
# # # # #         x = self.dropout(x)
# # # # #         x = self.linear2(x)
# # # # #         x = self.relu(x)
# # # # #         x = self.linear3(x)
# # # # #         x = self.relu(x)
# # # # #         x = self.linear4(x)
# # # # #         x = self.sig(x)
# # # # #         return x
# # # # # model = Model(30,1).to(DEVICE)

# # # # # criterion= nn.BCELoss()
# # # # # optimizer = optim.Adam(model.parameters(), lr = 0.01)

# # # # # def train(model, criterion, optimizer, loader):
# # # # #     total_loss = 0

# # # # #     for x_batch, y_batch in loader:
# # # # #         optimizer.zero_grad()
# # # # #         hypothesis = model(x_batch)
# # # # #         loss = criterion(hypothesis, y_batch)

# # # # #         loss.backward()
# # # # #         optimizer.step()

# # # # #         total_loss += loss.item()

# # # # #     return total_loss/len(loader)

# # # # # epochs = 300

# # # # # for epoch in range(1, epochs+1):
# # # # #     loss = train(model, criterion, optimizer, train_loader)
# # # # #     print('epochs:{}, loss:{}'.format(epoch, loss))



# # # # # def evaluate(model, criterion, loader):
# # # # #     model.eval()
# # # # #     total_loss = 0

# # # # #     for x_batch, y_batch in loader:
# # # # #         with torch.no_grad():
# # # # #             y_pred = model(x_batch)
# # # # #             loss2 = criterion(y_pred, y_batch)

# # # # #             total_loss += loss2.item()

# # # # #         return total_loss/len(loader)
    
# # # # # last_loss = evaluate(model, criterion, test_loader)

# # # # # print('최종 loss:', last_loss)

# # # # # y_predict = model(x_test)

# # # # # y_pred = np.round(y_predict.detach().cpu().numpy())


# # # # # y_true = y_test.detach().cpu().numpy()

# # # # # accuracy = accuracy_score(y_true, y_pred)
# # # # # r2 = r2_score(y_true, y_pred)

# # # # # print('accuracy score:', accuracy)


# # # # import numpy as np
# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.optim as optim

# # # # if torch.cuda.is_available():
# # # #     DEVICE = torch.device('cuda')
# # # # elif torch.backends.mps.is_available():
# # # #     DEVICE = torch.device('mps')
# # # # else:
# # # #     DEVICE = torch.device('cpu')
    
# # # # print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

# # # # x_train = np.array([1,2,3,4,5,6,7])
# # # # y_train = np.array([1,2,3,4,5,6,7])

# # # # x_test = np.array([8,9,10,11])
# # # # y_test = np.array([8,9,10,11])
# # # # x_pre = np.array([12,13,14])


# # # # x_train = torch.tensor(x_train, dtype= torch.float32).unsqueeze(1).to(DEVICE)
# # # # x_test = torch.tensor(x_test, dtype= torch.float32).unsqueeze(1).to(DEVICE)
# # # # x_pre = torch.tensor(x_pre, dtype= torch.float32).unsqueeze(1).to(DEVICE)
# # # # y_train = torch.tensor(y_train, dtype= torch.float32).unsqueeze(1).to(DEVICE)
# # # # y_test = torch.tensor(y_test, dtype= torch.float32).unsqueeze(1).to(DEVICE)

# # # # x_mean = torch.mean(x_train).to(DEVICE)
# # # # x_std = torch.std(x_train).to(DEVICE)

# # # # x_train = ((x_train-x_mean)/x_std).to(DEVICE)

# # # # model = nn.Sequential(
# # # #     nn.Linear(1, 16),
# # # #     nn.Linear(16,8),
# # # #     nn.Linear(8,1),

# # # # ).to(DEVICE)

# # # # criterion = nn.MSELoss()
# # # # optimizer= optim.Adam(model.parameters(), lr = 0.01)

# # # # def train(model, criterion, optimizer, x_train, y_train):
# # # #     optimizer.zero_grad()
# # # #     hypothesis = model(x_train)
# # # #     loss = criterion(hypothesis, y_train)

# # # #     loss.backward()
# # # #     optimizer.step()

# # # #     return loss.item()

# # # # epochs = 1000

# # # # for epoch in range(1, epochs+1):
# # # #     loss = train(model, criterion, optimizer, x_train, y_train)
# # # #     print('epoch :{}, loss:{}'.format(epoch, loss))

# # # # def evaluate(model, criterion, x_test, y_test):
# # # #     model.eval()
# # # #     with torch.no_grad():
# # # #         y_predict = model(x_test)
# # # #         loss2 = criterion(y_test, y_predict)
# # # #     return loss2.item()

# # # # loss2 = evaluate(model, criterion, x_test, y_test)

# # # # x_pred = (x_pre- x_mean)/x_std

# # # # result = model(x_pred)

# # # # print('xpred 예측값:', result.detach().cpu().numpy())

# # # import numpy as np
# # # import pandas as pd
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # from sklearn.datasets import load_breast_cancer
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.preprocessing import StandardScaler
# # # from sklearn.metrics import accuracy_score



# # # if torch.cuda.is_available():
# # #     DEVICE = torch.device('cuda')
# # # elif torch.backends.mps.is_available():
# # #     DEVICE = torch.device('mps')
# # # else:
# # #     DEVICE = torch.device('cpu')
    
# # # print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)



# # # import os

# # # if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
# # #     BASE_PATH = '/workspace/TensorJae/Study25/'
# # # else:                                                 # 로컬인 경우
# # #     BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
# # # basepath = os.path.join(BASE_PATH)

# # # #1 data
# # # path = basepath +  '_data/diabetes/'

# # # train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# # # test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# # # sample_submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col= 0)

# # # x = train_csv.drop(['Outcome'], axis=1)
# # # y = train_csv['Outcome']

# # # x = x.replace(0, np.nan)
# # # x = x.fillna(train_csv.mean())

# # # x_train, x_test, y_train, y_test = train_test_split(
# # #     x,y, train_size=0.8, shuffle=True, random_state=337, stratify=y
# # # )

# # # scaler = StandardScaler()
# # # x_train = scaler.fit_transform(x_train)
# # # x_test = scaler.transform(x_test)

# # # print(x_train.shape)
# # # print(y_train.shape)

# # # x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
# # # x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
# # # y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
# # # y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)


# # # model = nn.Sequential(
# # #     nn.Linear(8, 64),
# # #     nn.ReLU(),
# # #     nn.Linear(64, 32),
# # #     nn.ReLU(),
# # #     nn.Linear(32, 32),
# # #     nn.ReLU(),
# # #     nn.Linear(32, 16),
# # #     nn.SiLU(),
# # #     nn.Linear(16,1),
# # #     nn.Sigmoid()
# # # ).to(DEVICE)

# # # criterion = nn.BCELoss()
# # # optimizer = optim.Adam(model.parameters(), lr = 0.01)

# # # def train(model, criterion, optimizer, x, y):
# # #     optimizer.zero_grad()
# # #     hypothesis = model(x)
# # #     loss = criterion(hypothesis, y)
# # #     loss.backward()
# # #     optimizer.step()
    
# # #     return loss.item()

# # # epochs = 200

# # # for epoch in range(1, epochs+1):
# # #     loss = train(model, criterion, optimizer, x_train, y_train)
# # #     print('epochs:{}, loss:{}'.format(epoch, loss))

# # # def evaluate(model, criterion, x,y):
# # #     model.eval()

# # #     with torch.no_grad():
# # #         y_pred = model(x)
# # #         loss2 = criterion(y, y_pred)

# # #     return loss2.item()

# # # last_loss = evaluate(model, criterion, x_test, y_test)
# # # print('최종 Loss:', last_loss)

# # # y_predict = model(x_test)

# # # acc= accuracy_score(y_test.cpu().detach().numpy(), y_predict.cpu().detach().numpy().round())

# # # print('accuracy:', acc)



# # from sklearn.datasets import fetch_covtype
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.metrics import accuracy_score
# # import torch.nn as nn
# # import torch.optim as optim
# # import torch
# # import pandas as pd
# # import numpy as np
# # import random
# # # # # SEED = 42
# # # # # random.seed(SEED)
# # # # # np.random.seed(SEED)
# # # # # torch.manual_seed(SEED)

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



# # dataset = fetch_covtype()
# # x = dataset.data
# # y = dataset.target

# # print(dataset.target_names)
# # print(np.unique_counts(y))

# # print(x.shape, y.shape)


# # x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=SEED)

# # scaler = StandardScaler()
# # x_train = scaler.fit_transform(x_train)
# # x_test = scaler.transform(x_test)


# # x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
# # x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

# # y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
# # y_test = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

# # print(x_train.shape)
# # print(y_train.shape)

# # #2 model 
# # model = nn.Sequential(
# #     nn.Linear(54, 64),
# #     nn.ReLU(),
# #     nn.Linear(64, 32),
# #     nn.ReLU(),
# #     nn.Linear(32, 32),
# #     nn.ReLU(),
# #     nn.Linear(32, 16),
# #     nn.SiLU(),
# #     nn.Linear(16,7),
# #     # nn.Softmax() #sparse categorical entropy 를 했기 때문에 마지막 layer linear 로 하면 된다.
# # ).to(DEVICE)

# # criterion = nn.CrossEntropyLoss()
# # optimizer = optim.Adam(model.parameters(), lr = 0.01)

# # def train(model, criterion, optimizer, x,y):
# #     optimizer.zero_grad()
# #     hypothesis = model(x)
# #     loss = criterion(hypothesis, y)
# #     loss.backward()
# #     optimizer.step()

# #     return loss.item()

# # epochs= 1000

# # for epoch in range(1, epochs+1):
# #     loss = train(model, criterion, optimizer, x_train, y_train)
# #     print('epochs:{}, loss:{}'.format(epoch, loss))


# # def evaluate(model, criterion, x,y):
# #     model.eval()
# #     with torch.no_grad():
# #         y_pred = model(x)
# #         loss2 = criterion(y_pred, y)
# #     return loss2.item()

# # last_loss = evaluate(model, criterion, x_test, y_test)

# # print('final loss:', last_loss)

# # y_predict = model(x_test)
# # y_pred = y_predict.detach().cpu().numpy().argmax(axis=1)
# # y_pred = y_predict.detach().cpu().numpy().argmax(axis=1)
# # y_true = y_test.detach().cpu().numpy()

# # acc = accuracy_score(y_true, y_pred)
# # print('acc:' ,acc)


# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# import torch.nn as nn
# import torch.optim as optim

# import torch
# import random
# import numpy as np
# import pandas as pd

# SEED = 42

# random.seed(SEED)
# np.random.seed(SEED)


# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     DEVICE = torch.device('mps')
# else:
#     DEVICE = torch.device('cpu')
    
# print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)


# dataset= load_digits()
# x = dataset.data
# y = dataset.target

# print(np.unique_counts(y))

# x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=SEED)



# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = torch.tensor(x_train, dtype= torch.float32).to(DEVICE)
# x_test = torch.tensor(x_test, dtype= torch.float32).to(DEVICE)
# y_train = torch.tensor(y_train, dtype= torch.long).to(DEVICE)
# y_test = torch.tensor(y_test, dtype= torch.long).to(DEVICE)
# print(x_train.shape)
# print(y_train.shape)



# class Model(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Model, self).__init__()
        
#         self.linear1 = nn.Linear(input_dim, 64) 
#         self.linear2 = nn.Linear(64, 32)
#         self.linear3 = nn.Linear(32, 32)
#         self.linear4 = nn.Linear(32,16)
#         self.linear5 = nn.Linear(16, output_dim)
#         self.relu = nn.ReLU()
#         self.silu = nn.SiLU()
#         self.dropout = nn.Dropout(0.2)
        
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.linear2(x)
#         x = self.relu(x)
#         x = self.linear3(x)
#         x = self.relu(x)
#         x = self.linear3(x)
#         x = self.relu(x)
#         x = self.linear3(x)
#         x = self.relu(x)
#         x = self.linear4(x)
#         x = self.silu(x)
#         x = self.linear5(x)
#         return x
    

# model = Model(64,10).to(DEVICE) 


# criterion= nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr =0.01)

# def train(model, criterion, optimizer, x,y):
#     optimizer.zero_grad()
#     hypothesis = model(x)
#     loss = criterion(hypothesis, y)
#     loss.backward()
#     optimizer.step()
#     return loss.item()

# epochs = 100

# for epoch in range(1, epochs+1):
#     loss = train(model, criterion, optimizer, x_train, y_train)
#     print('epoch:{}, loss:{}'.format(epoch, loss))

# def evaluate(model, criterion, x,y):
#     model.eval()

#     with torch.no_grad():
#         y_pred = model(x)
#         loss2 = criterion(y_pred, y)

#     return loss2.item()

# last_loss = evaluate(model, criterion, x_test, y_test)

# print('last loss:', last_loss)


# y_predict = model(x_test)
# y_pred = y_predict.detach().cpu().numpy().argmax(axis=1)
# y_true = y_test.detach().cpu().numpy()

# acc = accuracy_score(y_true, y_pred)

# print('acc:', acc)


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
    
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

# 1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
 
x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9],])        # (7, 3)
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)



x = x.reshape(x.shape[0], x.shape[1], 1)
x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE)

from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x,y)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

aaa = iter(train_loader)
bbb = next(aaa)


exit()

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_layer1 = nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )