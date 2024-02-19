import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
from torch import nn, optim
import tkinter as tk
import matplotlib.pyplot as plt

# Определение автоэнкодера
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, k):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, k),  # Используем k в качестве размерности скрытого пространства
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(k, hidden_dim),  # Используем k в качестве размерности скрытого пространства
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Загрузка набора данных Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Нормализация данных
sc = StandardScaler()
X = sc.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0, stratify=y)

# Преобразование данных в тензоры PyTorch
X_train = torch.tensor(X_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)

# Создание и обучение автоэнкодера
k = 256 # Вы можете выбрать любое значение k, которое соответствует вашим требованиям
model = Autoencoder(X_train.shape[1], 256, k)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):  # 100 эпох для примера, возможно, вам потребуется больше
    output = model(X_train[y_train == 0])  # Обучение только на "нормальных" образцах
    loss = criterion(output, X_train[y_train == 0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Loss: {loss.item()}')

# Проверка работы автоэнкодера
outputs = model(X_test)
mse = np.mean(np.power(X_test.detach().numpy() - outputs.detach().numpy(), 2), axis=1)
roc_auc = roc_auc_score(y_test > 0, mse)
print(f'ROC AUC: {roc_auc}')

# Визуализация MSE
plt.figure(figsize=(10, 10))
colors = ['blue' if mse_value <= 2 else 'red' for mse_value in mse]  # Используем MSE для определения цвета
plt.scatter(range(len(mse)), mse, c=colors)
plt.xlabel('Наблюдение')
plt.ylabel('MSE')
plt.show()
