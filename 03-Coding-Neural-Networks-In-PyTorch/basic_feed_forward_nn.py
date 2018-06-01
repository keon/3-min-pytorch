import torch
import numpy
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plot
import torch.nn.functional as F

##인공신경망을 이용해 간단한 분류 모델 구현하기
#분류 설명

# #Define Data Set in Numpy------------------------------------------------------------------------------------------------------
# 인공신경망을 구현하고 학습시키기 전에, 학습에 쓰일 데이터들을 Numpy 라이브러리를 사용해 만들어 보겠습니다. 

def label_map(y_, from_, to_):
    y = numpy.copy(y_)
    for f in from_:
        y[y_ == f] = to_
    return y
  
n_dim = 2
x_tra, y_tra = make_blobs(n_samples=80, n_features=n_dim, centers=[[1,1],[-1,-1],[1,-1],[-1,1]], shuffle=True, cluster_std=0.3)
x_tes, y_tes = make_blobs(n_samples=20, n_features=n_dim, centers=[[1,1],[-1,-1],[1,-1],[-1,1]], shuffle=True, cluster_std=0.3)
y_tra = label_map(y_tra, [0, 1], 0)
y_tra = label_map(y_tra, [2, 3], 1)
y_tes = label_map(y_tes, [0, 1], 0)
y_tes = label_map(y_tes, [2, 3], 1)

# 위 코드를 통해서 원소가 두개 있는 벡터 형태의 데이터 들을 만들었습니다. 코드에서 볼 수 있듯이,
# 트레이닝 데이터(training data set) 에는 80개의 데이터가 있고, 테스트 데이터(Test Data set) 에는
# 20개의 데이터가 있는 것을 확인 하실 수 있습니다. 데이터가 [1,1]벡터 혹은 [-1,-1]벡터 와 가까이 있으면 0 이라고 레이블링 해줬고,
# [1,-1]벡터 혹은 [-1,1]벡터 가까이 있으면 반대로 1 이라고 레이블링 해 주었습니다. 이러한 패턴을 보이는 
# 데이터를 통틀어 'XOR 패턴의 데이터'라고 합니다. 이번 장에서 우리의 목표는 XOR 패턴의 데이터를 분류하는 간단한 인공신경망을 구현해 보는 겁니다.
#------------------------------------------------------------------------------------------------------------------------------
#For Data visualizeation purpose---------------------------------------------------------------------------------------------
#Matplotlib 라이브러리를 사용해 밑의 코드를 실행시켜 보면 데이터가 어느 패턴을 보이는지 한 눈에 볼 수 있습니다.
def vis_data(x,y = None, c = 'r'):
	if y is None:
		y = [None] * len(x)
	for x_, y_ in zip(x,y):
		if y_ is None:
			plot.plot(x_[0], x_[1], '*',markerfacecolor='none', markeredgecolor=c)
		else:
			plot.plot(x_[0], x_[1], c+'o' if y_ == 0 else c+'+')

plot.figure()
vis_data(x_tra, y_tra, c='r')
plot.show()
#------------------------------------------------------------------------------------------------------------------------------
#Turn Data Set to Pytorch tensors
#모델을 구현하기 전, 위에서 정의한 데이터들을 넘파이 리스트가 아닌 파이토치 텐서로 재정의 합니다.
x_tra = torch.FloatTensor(x_tra)
x_tes = torch.FloatTensor(x_tes)
y_tra = torch.LongTensor(y_tra)
y_tes = torch.LongTensor(y_tes)
#-------------------------------------------------------------------------
#Our first Neural Network Model
# 자, 그럼 매우 간단한 인공신경망을 구현해 보겠습니다. 파이토치에서는 인공신경망을 하나의 파이썬 객체(Object)로 나타낼 수 있습니다.
# __init__()에선 인공신경망 속에 필요한 행렬곱, 활성화 함수, 그리고 그 외 다른 계산식들을 함수로 정의합니다.
# forward()에선 앞의 __init__()에서 정의한 함수들을 호출하여 입력된 데이터에 대한 결과값을 출력합니다.

class Feed_forward_nn(torch.nn.Module):
		def __init__(self, input_size, hidden_size):
			super(Feed_forward_nn, self).__init__()
			self.input_size = input_size
			self.hidden_size  = hidden_size
			self.linear_1 = torch.nn.Linear(self.input_size, self.hidden_size)
			self.relu = torch.nn.ReLU()
			self.linear_2 = torch.nn.Linear(self.hidden_size, 2)
		def forward(self, input_tensor):
			linear1 = self.linear_1(input_tensor)
			relu = self.relu(linear1)
			linear2 = self.linear_2(relu)
			# output = self.sigmoid(linear2)
			return linear2

#Train our Model
# 자, 이제 학습시킬 인공신경망도 있으니 학습에 필요한 러닝레이트(Learning Rate),
# 오차(Loss), 이포씨(Epoch), 그리고 최적화 알고리즘(Optimizer)을 정의합니다. 
model = Feed_forward_nn(2, 5)
learning_rate = 0.03
# criterion = F.cross_entropy()
epochs = 1000
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
# 러닝레이트는 쉽게 말해 얼마나 급하게 학습을 시키고 싶은지
# 정해주는 값이라고 할 수 있습니다. 너무 크게 값을 설정해 버리면 모델이 오차의 최소점을 지나치게 되고, 값이 너무 작으면
# 학습이 느려집니다.
# 알맞는 러닝레이트틀 선택하기 위해선 랜덤서치(Random Search) 와 같은 알고리즘을 사용해야 하며, 최적화된 
# 러닝레이트를 찾아내는 과정은 현재 딥러닝 학계에서도 활발히 연구되고 있는 주제입니다.
# 다음으로 오차 함수(loss function) 를 정의합니다. 다시 한번 말씀드리지만, 오차 함수는 머신러닝 모델의 결과값과 실제답과의 차이를 산술적으로 표현한 함수입니다.
# 이번 예제에서는 예측답과 실제답의 사이의 확률분포의 차이를 나타내 주는
# 교차 엔트로피(Cross Entrophy) 라는 오차함수를 사용하겠습니다.
# # 이포씨(epoch)
# 는 쉽게 말해 총 몇번 학습을 시키고 싶은지 정해주는 값입니다. 즉, 위의 코드에선 '총 1000 번 오차의 최소값의 방향으로
# 움직이겠다' 라고 선언한 것입니다.

#Performance of the model before training
model.eval()
test_loss_before = F.cross_entropy(torch.squeeze(model(x_tes) ), y_tes)
print('Before Training, test loss is ', test_loss_before.item())

for epoch in range(epochs):
	model.train()
	optimizer.zero_grad()
	train_output = model(x_tra)
	# train_output = torch.squeeze(train_output)
	train_loss = F.cross_entropy(train_output, y_tra)
	train_loss.backward()
	optimizer.step()

#Performance of the model before training
model.eval()
test_loss = F.cross_entropy(torch.squeeze(model(x_tes) ), y_tes) 
print('After Training, test loss is ', test_loss.item())

#Additional model test
inf_tensor = torch.FloatTensor([1,1])
inf_tensor2 = torch.FloatTensor([-1,-1])
inf_tensor3 = torch.FloatTensor([1,-1])
inf_tensor4 = torch.FloatTensor([-1,1])
print(model(inf_tensor))
print(model(inf_tensor2))
print(model(inf_tensor3))
print(model(inf_tensor4))
