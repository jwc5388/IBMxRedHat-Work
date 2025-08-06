import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import os
from datetime import datetime
import pennylane as qml

# 경로
base_path = '/Users/jaewoo000/Desktop/IBM:RedHat/Study25/_data/quantum/'
model_path = os.path.join(base_path, 'model_20250805_181637_final_best_train_acc_92.6667_improved.pth')

# 디바이스
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# QNN 디바이스 정의
dev = qml.device("default.qubit", wires=5)

# 양자 회로 정의
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    num_qubits = 5
    layers = 2
    for l in range(layers):
        for i in range(num_qubits):
            qml.RX(inputs[:, i % inputs.shape[1]], wires=i)
            qml.RY(weights[(l * num_qubits + i) % weights.shape[0]], wires=i)
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[num_qubits - 1, 0])
    for i in range(num_qubits):
        qml.RZ(weights[(i + weights.shape[0] // 2) % weights.shape[0]], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

# 모델 정의
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 5)
        self.norm = nn.LayerNorm(5)
        self.q_params = nn.Parameter(torch.rand(30))
        self.fc2 = nn.Linear(5, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.norm(x)
        q_out = quantum_circuit(x, self.q_params)
        q_out = torch.stack(list(q_out), dim=1).to(torch.float32)
        x = self.fc2(q_out)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# 데이터셋 로딩 및 필터링
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset_full = datasets.FashionMNIST(root=base_path, train=False, download=True, transform=transform)
indices_test = [i for i, (_, label) in enumerate(test_dataset_full) if label in [0, 6]]
test_subset = Subset(test_dataset_full, indices_test)

# 라벨 6 → 1로 매핑
for i in range(len(test_subset.dataset.targets)):
    if test_subset.dataset.targets[i] == 6:
        test_subset.dataset.targets[i] = 1

test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

# 모델 로드
model = HybridModel().to(device)
# model.load_state_dict(torch.load(model_path))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# 추론 및 정확도 평가
correct, total = 0, 0
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        pred = output.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(pred.cpu().numpy())

acc = correct / total * 100
print(f"✅ 재현된 정확도: {acc:.2f}%")

# 제출 CSV 생성
final_preds = [0 if p == 0 else 6 for p in all_preds]
now = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_path = f"{base_path}y_pred_{now}_reproduced.csv"
pd.DataFrame({"y_pred": final_preds}).to_csv(csv_path, index=False, header=False)
print(f"✅ 예측 결과 저장 완료: {csv_path}")