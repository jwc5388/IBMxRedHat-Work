
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
    
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)


import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

# 1. Load Data
path = basepath + '_data/kaggle/bank/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. Encode categorical features
le_geo = LabelEncoder()
le_gender = LabelEncoder()

train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])

test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

# 3. Drop unneeded columns
train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# 4. Separate features and target
x = train_csv.drop(['Exited'], axis=1)
feature_names = x.columns
y = train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.8, shuffle=True, random_state=337, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train.to_numpy()).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test.to_numpy()).unsqueeze(1).to(DEVICE)

print(x_train.shape)
print(y_train.shape)

# exit()



class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, 64) 
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32,16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(0.2)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.silu(x)
        x = self.linear5(x)
        x = self.sig(x)
        return x
    
    
    
model = Model(10,1).to(DEVICE)


# #2 model 
# model = nn.Sequential(
#     nn.Linear(13, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.SiLU(),
#     nn.Linear(16,3),
#     # nn.Softmax() #sparse categorical entropy 를 했기 때문에 마지막 layer linear 로 하면 된다.
# ).to(DEVICE)


criterion = nn.BCELoss()   #sparse categorical entropy = onehot 안해줘도 됨
optimizer = optim.Adam(model.parameters(), lr = 0.005)







def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss= criterion(hypothesis,y)
    
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 1000

for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epochs:{}, loss:{}'.format(epoch, loss)) #verbose

print('==================================')

def evaluate(model, criterion, x,y):
    model.eval()
    
    with torch.no_grad():
        y_pred = model(x)
        loss2 = criterion(y_pred, y)
        
    return loss2.item()

last_loss = evaluate(model, criterion, x_test, y_test)

print('최종 Loss:', last_loss)


y_predict = model(x_test)
y_pred = y_predict.detach().cpu().numpy().argmax(axis=1)
y_true = y_test.detach().cpu().numpy()
acc = accuracy_score(y_true, y_pred)
print('accuracy score:', acc)

# 최종 Loss: 0.3254335820674896
# accuracy score: 0.7884085194049747