
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
import torch

# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     DEVICE = torch.device('mps')
# else:
#     DEVICE = torch.device('cpu')
    
# print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
# ❌ elif torch.backends.mps.is_available():
# ❌     DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

#1 data
dataset = load_iris()
x = dataset.data
y = dataset.target

print(dataset.target_names)  # ['setosa' 'versicolor' 'virginica']
print(len(set(dataset.target)))  # 3

x_train, x_test , y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=42)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)


#다양한 플랫폼 (특히gpu) 에서 타입 충돌을 피하기 위해
#pytorch는 target label은 무조건 int64(long) 로 고정했습니다.
print(x_train.shape)
print(y_train.shape)

# exit()


#2 model 
model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.SiLU(),
    nn.Linear(16,3),
    # nn.Softmax() #sparse categorical entropy 를 했기 때문에 마지막 layer linear 로 하면 된다.
).to(DEVICE)


criterion = nn.CrossEntropyLoss()   #sparse categorical entropy = onehot 안해줘도 됨
optimizer = optim.Adam(model.parameters(), lr = 0.005)







def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss= criterion(hypothesis,y)
    
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 200

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

