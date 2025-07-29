
from sklearn.datasets import load_iris, load_wine, fetch_covtype, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
import torch

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
    
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)


# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda')
# # ❌ elif torch.backends.mps.is_available():
# # ❌     DEVICE = torch.device('mps')
# else:
#     DEVICE = torch.device('cpu')

#1 data
dataset = load_digits()
x = dataset.data
y = dataset.target

print(dataset.target_names)  
print(len(set(dataset.target)))  #10

x_train, x_test , y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=42)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# x_train = torch.FloatTensor(x_train).to(DEVICE)
# x_test = torch.FloatTensor(x_test).to(DEVICE)

# y_train = torch.LongTensor(y_train).to(DEVICE)
# y_test = torch.LongTensor(y_test).to(DEVICE)


x_train = torch.tensor(x_train, dtype= torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype= torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype= torch.long).to(DEVICE)
y_test = torch.tensor(y_test, dtype= torch.long).to(DEVICE)
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
        return x
        
model = Model(64,10).to(DEVICE) 


# #2 model 
# model = nn.Sequential(
#     nn.Linear(64, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 32),
#     nn.ReLU(),
#     nn.Linear(32, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.SiLU(),
#     nn.Linear(16,10),
#     # nn.Softmax() #sparse categorical entropy 를 했기 때문에 마지막 layer linear 로 하면 된다.
# ).to(DEVICE)


criterion = nn.CrossEntropyLoss()   #sparse categorical entropy = onehot 안해줘도 됨
optimizer = optim.Adam(model.parameters(), lr = 0.01)







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

print('acc:' ,acc)


# 최종 Loss: 0.3160257637500763
# acc: 0.9611111111111111