import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
    
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)


x = np.array(range(100))
y = np.array(range(1,101))
x_pred = np.array([101,102])


x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, train_size=0.8)


x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
print(x_train.shape)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
x_pred = torch.FloatTensor(x_pred).unsqueeze(1).to(DEVICE)

y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

x_mean = torch.mean(x_train)
x_std = torch.std(x_train)

x_train = (x_train - torch.mean(x_train))/ torch.std(x_train)



model = nn.Sequential(
    nn.Linear(1, 16),
    nn.Linear(16, 16),
    nn.Linear(16, 8),
    nn.Linear(8,1)
).to(DEVICE)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)



def train(model, criterion,optimizer, x_train, y_train):
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 5000

for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train,y_train)
    print('epoch: {}, loss: {}'.format(epoch, loss))

def evaluate(model, criterion, x_test, y_test):
    model.eval()
    with torch.no_grad():
        y_predict = model(x_test)
        loss2 = criterion(y_test, y_predict)
        
    return loss2.item()

loss2 = evaluate(model, criterion,x_test, y_test)

print('final loss:', loss2)

x_pred1 = (x_pred - x_mean)/x_std

result = model(x_pred1)
print('x_pred 예측값:', result.detach().cpu().numpy())
