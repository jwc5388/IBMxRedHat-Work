import torch.nn as nn

# 1. 기본 선형 레이어 (Fully Connected)
nn.Linear(in_features, out_features)

# 2. CNN 관련 레이어
nn.Conv1d(in_channels, out_channels, kernel_size)
nn.Conv2d(in_channels, out_channels, kernel_size)
nn.Conv3d(...)
nn.MaxPool1d(kernel_size)
nn.MaxPool2d(kernel_size)
nn.AvgPool2d(kernel_size)
nn.AdaptiveAvgPool2d((1, 1))

# 3. RNN 계열 레이어
nn.RNN(input_size, hidden_size)
nn.LSTM(input_size, hidden_size)
nn.GRU(input_size, hidden_size)

# 4. 정규화 (Normalization)
nn.BatchNorm1d(num_features)
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)

# 5. 활성화 함수 (Activation)
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=1)
nn.LeakyReLU(negative_slope=0.01)

# 6. 드롭아웃 (Dropout)
nn.Dropout(p=0.2)

# 7. Embedding / NLP / Transformer
nn.Embedding(num_embeddings, embedding_dim)
nn.TransformerEncoderLayer(d_model, nhead)
nn.MultiheadAttention(embed_dim, num_heads)

# 8. 기타 레이어 및 유틸
nn.Flatten()
nn.Identity()
nn.Upsample(scale_factor=2)
nn.Sequential(...)  # 여러 레이어 묶을 때

# 9. 입력 변경 관련 (forward 안에서 자주 사용)
x.view(x.size(0), -1)           # Flatten
torch.flatten(x, start_dim=1)   # Flatten 대체 함수


# Binary
y_pred = (model(x) > 0.5).float()

# Multiclass
y_pred = torch.argmax(model(x), dim=1)

# Regression
y_pred = model(x)  # 그대로 사용

# Multi-label
y_pred = (model(x) > 0.5).float()