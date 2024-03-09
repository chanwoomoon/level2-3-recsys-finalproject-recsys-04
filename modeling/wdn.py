import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using {device} ")


df = pd.read_csv('/home/siyun/ephemeral/WDN/data/gift_data.csv')


# 각 product_id 별로 size(count) 계산

df['category_3'].fillna('unknown', inplace=True)
product_category_counts = df.groupby(['product_id', 'category_1', 'category_2', 'category_3']).size().reset_index(name='counts')

# 각 product_id 내에서 카테고리가 선택될 확률 계산
product_category_counts['total_counts'] = product_category_counts.groupby('product_id')['counts'].transform('sum')
product_category_counts['prob'] = product_category_counts['counts'] / product_category_counts['total_counts']

# 각 product_id에 대한 엔트로피를 계산
# -sigma(pi) * log2(pi)
product_category_counts['personality'] = -product_category_counts['prob'] * np.log2(product_category_counts['prob'])
product_entropy = product_category_counts.groupby('product_id')['personality'].sum().reset_index()

# 매핑
mean_entropy = product_entropy['personality'].mean()

df = df.merge(product_entropy, on='product_id', how='left')

df['personality'].fillna(mean_entropy, inplace=True)


category_cols = ['category_1', 'category_2', 'category_3', 'brand']
label_encoders = {}
for col in category_cols:
    unique_values = df[col].nunique()
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df = df.drop(['product_name','image_url','product_url'], axis=1)

from sklearn.preprocessing import StandardScaler

s1 = StandardScaler()
s2 = StandardScaler()

s1.fit(df['price'].values.reshape(-1, 1))
s2.fit(df['num_review'].values.reshape(-1, 1))

price = s1.transform(df['price'].values.reshape(-1, 1))
num_review = s2.transform(df['num_review'].values.reshape(-1, 1))

df['price_scaling'] =price
df['review_scaling'] = num_review

feature_cols = ['brand', 'category_1', 'category_2', 'category_3', 'rating', 'review_scaling', 'price_scaling', 'personality']
target_col = 'product_id'

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(df[feature_cols], df[target_col], test_size=0.2, random_state=42)



class GiftDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.tensor(self.features.iloc[idx].values, dtype=torch.float)  # 실수형 데이터를 처리하기 위해 dtype을 torch.float으로 변경
        if self.labels is not None:
            labels = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
            return features, labels
        return features

# feature_fields = ['rating', 'num_review', 'personality'] + category_cols
train_dataset = GiftDataset(X_train, y_train)
val_dataset = GiftDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


for batch in train_loader:
    features, labels = batch
    print(features)
    print(labels)
    print("Features shape:", features.shape)
    print("Features dtype:", features.dtype)
    print("Labels shape:", labels.shape)
    print("Labels dtype:", labels.dtype)
    break  # 첫 배치만 확인하고 반복문 탈출

next(iter(train_loader))

for features, labels in train_loader:
    print("Features max:", features.max(dim=0)[0])
    print("Features min:", features.min(dim=0)[0])
    print(labels)
    break  # 첫 배치에 대해서만 확인

import numpy as np 
import torch 
import torch.nn as nn 

# FM모델 등에서 활용되는 선형 결합 부분을 정의합니다.
class FeaturesLinear(nn.Module):
    def __init__(self, field_dims: np.ndarray, output_dim: int=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


# factorization을 통해 얻은 feature를 embedding 합니다.
class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)
# NCF 모델은 MLP와 GMF를 합하여 최종 결과를 도출합니다.
# MLP을 구현합니다.
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)


    def forward(self, x):
        return self.mlp(x)

class WideAndDeepModel(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims=field_dims, output_dim=1)
        self.embedding = nn.ModuleList([nn.Embedding(num_embeddings=dim, embedding_dim=embed_dim) for dim in field_dims])
        self.mlp = nn.Sequential(
            nn.Linear(sum(field_dims) * embed_dim, mlp_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            *[nn.Sequential(nn.Linear(mlp_dims[i], mlp_dims[i+1]), nn.ReLU(), nn.Dropout(dropout)) for i in range(len(mlp_dims)-1)],
            nn.Linear(mlp_dims[-1], 1)
        )

    def forward(self, x):
        # 임베딩된 특성을 연결합니다.
        embedded_features = [embedding(x[:, i].unsqueeze(-1)) for i, embedding in enumerate(self.embedding)]
        embedded = torch.cat(embedded_features, dim=2).view(x.size(0), -1)
        # 선형 부분의 출력을 계산합니다.
        x_linear = self.linear(x)
        # MLP 부분의 출력을 계산합니다.
        x_mlp = self.mlp(embedded)
        # 최종 출력을 계산합니다.
        output = x_linear + x_mlp
        return output.squeeze(1)

field_dims=[len(df[col].unique()) for col in feature_cols]
field_dims

import torch.optim as optim
from torch.nn import MSELoss

model = WideAndDeepModel(field_dims=[len(df[col].unique()) for col in feature_cols], embed_dim=10, mlp_dims=[64, 32], dropout=0.1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = MSELoss()



model = model.to(device)

# 학습
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        print(f'Features Shape: {features.shape}')
        print(f'Labels Shape: {labels.shape}')
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')

# 추천 함수
def recommend_topk(model, features, k=10):
    model.eval()
    with torch.no_grad():
        predictions = model(features)
        _, topk_indices = torch.topk(predictions, k)
    return topk_indices


