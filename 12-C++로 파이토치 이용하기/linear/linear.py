import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 하이퍼파라미터
input_size = 1
output_size = 1 
num_epochs = 60
learning_rate = 0.001

# 미니 데이터셋
x_train = torch.Tensor([
    [2.1], [3.3], [3.6], [4.4], [5.5],
    [6.3], [6.5], [7.0], [7.5], [9.7],
])
y_train = torch.Tensor([
    [1.0], [1.2], [1.9], [2.0], [2.5],
    [2.5], [2.2], [2.7], [3.0], [3.6],
])
dataset = TensorDataset(x_train, y_train)
data_loader = DataLoader(dataset)


# 선형회귀 모델
model = torch.nn.Linear(input_size, output_size)

# 오차 함수와 최적화 함수 
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 학습 시작
for epoch in range(num_epochs):
    for example, label in data_loader:
        prediction = model.forward(example)
        loss = criterion(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 학습 결과 그래프 그리기
predicted = model(x_train).detach()
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predicted.numpy(), label='Fitted line')
plt.legend()
plt.show()


