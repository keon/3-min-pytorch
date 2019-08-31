#!/usr/bin/env python
# coding: utf-8

# # 9.1 GAN으로 새로운 패션아이템 생성하기
# *GAN을 이용하여 새로운 패션 아이템을 만들어봅니다*
# GAN을 구현하기 위해 그 구조를 더 자세히 알아보겠습니다.
# GAN은 생성자(Generator)와 판별자(Discriminator) 2개의 신경망으로 이루어져 있습니다.
# 생성자는 실제 데이터와 비슷한 가짜 데이터를 만들어냅니다. 생성자가 만든 가짜 데이터는 '가짜' 라는 레이블을 부여받고 
# Fashion MNIST의 이미지와 같은 '진짜' 데이터와 함께 판별자에 입력됩니다.  
# 그러면 판별자는 진짜와 가짜 데이터를 구분하는 능력을 학습합니다. 여기서 재밌는점은 판별자가 가짜와 진짜를 제대로 분류할 때 마다 생성자에 대한 페널티는 늘어난다는 것입니다.
# 그러므로 생성자는 판별자가 좋은 퍼포먼스를 내는것을 방해하기 위해 더 진짜 데이터와 비슷한 데이터를 생성하게 됩니다.
# 이처럼 GAN은 이름 그대로 판별자와 생성자의 경쟁을 통해서 학습하는 모델입니다.

# ## GAN 구현하기

# 지금까지 해온 것 처럼 구현에 필요한 라이브러리들을 임포트합니다.

import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt


# 생성자는 랜덤한 텐서를 입력받아 기존 데이터와 비슷한 데이터를 창작하는 '신경망' 입니다. 그러므로 생성자에 입력되는 랜덤 텐서가 어떻게 설정되느냐에 따라 같은 코드라도 결과물과 퍼포먼스 근소하게 달라질 여지가 있습니다. 그러므로 여러분들이 직접 이 책의 GAN 코드를 보면서 구현한 결과와 책에서 보여주는 결과를 최대한 비슷하게 만들어주기 위해 학습 도중 생성되는 모든 랜덤한 값을 동일하게 설정해 주겠습니다.

torch.manual_seed(1)    # reproducible


# EPOCHS 과 BATCH_SIZE 등 학습에 필요한 하이퍼 파라미터 들을 설정해 줍니다.

# Hyper Parameters
EPOCHS = 100
BATCH_SIZE = 100
USE_CDA = torch.cuda.is_available()
DEVICE = -1#torch.device("cuda" if USE_CUDA else "cpu")
print("Using Device:", DEVICE)


# 학습에 필요한 데이터셋을 로딩합니다. 

# Fashion MNIST digits dataset
trainset = datasets.FashionMNIST('./.data',
    train=True,
    download=True,
    transform=transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
    ]))
train_loader = torch.utils.data.DataLoader(
    dataset     = trainset,
    batch_size  = BATCH_SIZE,
    shuffle     = True)


# 데이터의 로딩이 끝났으면 GAN의 생성자와 판별자를 구현합니다.  
# 지금까지는 신경망 모델들을 파이썬의 객체로써 정의해 주었습니다. 그렇게 함으로써 신경망의 복잡한 기능과 동작들을 함수의 형태로 편리하게 정의해 줄 수 있었습니다.
# 그러나 이번 예제에서 구현할 생성자와 판별자는 비교적 단순한 신경망이므로, 좀 더 간소한 방법을 이용해 정의해 보겠습니다.  
# Pytorch가 제공하는 Sequential 자료구조는 신경망의 forward() 동작에 필요한 동작들을 입력받아 이들을 차례대로 실행시키는 신경망 구조체를 만들어 줍니다.
# 생성자는 64차원의 랜덤한 텐서를 입력받아 이에 행렬곱(Linear)과 활성화 함수(ReLU, Tanh) 연산을 실행합니다. 생성자의 결과값은 784차원, 즉 Fashion MNIST 속의 이미지와 같은 차원의 텐서입니다.

# Generator 
G = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 784),
        nn.Tanh())


# 판별자는 784차원의 텐서를 입력받습니다. 판별자 역시 입력된 데이터에 행렬곱과 활성화 함수를 실행시키지만, 생성자와 달리 판별자의 결과값은 입력받은 텐서가 진짜 Fashion MNIST 데이터일 확률값입니다.

# Discriminator
D = nn.Sequential(
        nn.Linear(784, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid())


# 생성자와 판별자 학습에 쓰일 오차 함수와 최적화 알고리즘도 정의해 줍니다.


# Device setting
# D = D.to(DEVICE)
# G = G.to(DEVICE)

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)


# 모델 학습에 필요한 준비는 끝났습니다. 그럼 본격적으로 GAN을 학습시키는 loop을 만들어 보겠습니다. 

total_step = len(train_loader)
for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(train_loader):
        images = images.reshape(BATCH_SIZE, -1)#.to(-1)


# 데이터셋 속의 진짜 이미지에는 '진짜' 라는 레이블을, 반대로 생성자가 만든 이미지에는 '가짜'라는 레이블링을 해 줘야 합니다. 이 두 레이블을 나타내는 레이블 텐서를 정의해 줍니다.

real_labels = torch.ones(BATCH_SIZE, 1)#.to(-1)
fake_labels = torch.zeros(BATCH_SIZE, 1)#.to(-1)


# 판별자는 실제 이미지를 보고 '진짜'라고 구분짓는 능력을 학습해야 합니다. 그러기 위해선 실제 이미지를 판별자 신경망에 입력시켜 얻어낸 결과값과 '진짜' 레이블 간의 오차값을 계산해야 합니다.

outputs = D(images)
d_loss_real = criterion(outputs, real_labels)
real_score = outputs


# 다음으로는 생성자의 동작을 정의합니다. 생성자는 무작위한 텐서를 입력받아 실제 이미지와 같은 차원의 텐서를 배출해야합니다.

z = torch.randn(BATCH_SIZE, 64)#.to(-1)
fake_images = G(z)


# 생성자가 만들어낸 fake_images를 판별자에 입력합니다. 이번엔 결과값과 '가짜' 레이블 간의 오차를 계산해야 합니다.

outputs = D(fake_images)
d_loss_fake = criterion(outputs, fake_labels)
fake_score = outputs


# 실제 데이터와 가짜 데이터를 가지고 낸 오차를 더해줌으로써 판별자 신경망의 전체 오차가 계산됩니다.
# 그 다음 과정은 역전파 알고리즘과 경사 하강법을 통하여 판별자 신경망을 학습시키는 겁니다.

d_loss = d_loss_real + d_loss_fake
d_optimizer.zero_grad()
d_loss.backward()
d_optimizer.step()


# 판별자를 학습시키는 코드를 모두 작성했으면 이제 생성자를 학습시킬 차례입니다.  
# 생성자가 더 진짜같은 데이터셋을 만들어내려면, 생성자가 만들어낸 가짜 이미지를 판별자가 진짜 라고 착각하게 만들어야 합니다.  
# 즉, 생성자의 결과물을 다시 판별자에 입력시켜, 그 결과물과 real_labels간의 오차를 최소화 시키는 식으로 학습을 진행해야 합니다.

fake_images = G(z)
outputs = D(fake_images)
g_loss = criterion(outputs, real_labels)


# 그리고 마찬가지로 경사 하강법과 역전파 알고리즘을 사용해서 모델의 학습을 완료합니다.

d_optimizer.zero_grad()
g_optimizer.zero_grad()
g_loss.backward()
g_optimizer.step()


# 학습을 진행하는 동안 오차를 확인하고 생성자의 결과물을 시각화하는 코드 또한 추가시켰습니다.

if (i+1) % 200 == 0:
    print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
          .format(epoch, EPOCHS, i+1, total_step, d_loss.item(), g_loss.item(), 
                  real_score.mean().item(), fake_score.mean().item()))
    
if (epoch+1) % 10 == 0 and (i+1) % 100 == 0 :
    fake_images = np.reshape(fake_images.data.numpy()[0],(28, 28))
    plt.imshow(fake_images, cmap = 'gray')
    plt.show()


# 학습이 끝난 생성자의 결과물을 한번 확인해 보겠습니다.

# ![generated_image0](./assets/generated_image0.png)
# ![generated_image1](./assets/generated_image1.png)
# ![generated_image2](./assets/generated_image2.png)
# ![generated_image3](./assets/generated_image3.png)
# ![generated_image4](./assets/generated_image4.png)













# 전체 코드

import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np


torch.manual_seed(1)    # reproducible


# Hyper Parameters
EPOCHS = 100
BATCH_SIZE = 100
USE_CDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

print("Using Device:", DEVICE)


# Fashion MNIST digits dataset
trainset = datasets.FashionMNIST('./.data',
    train=True,
    download=True,
    transform=transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
    ]))
train_loader = torch.utils.data.DataLoader(
    dataset     = trainset,
    batch_size  = BATCH_SIZE,
    shuffle     = True)


# Discriminator
D = nn.Sequential(
        nn.Linear(784, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid())


# Generator 
G = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 784),
        nn.Tanh())



# Device setting
D = D.to(DEVICE)
G = G.to(DEVICE)

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)


total_step = len(train_loader)
for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(train_loader):
        images = images.reshape(BATCH_SIZE, -1).to(DEVICE)
        
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE)
        fake_labels = torch.zeros(BATCH_SIZE, 1).to(DEVICE)

        # Train Discriminator

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(BATCH_SIZE, 64).to(DEVICE)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator

        # Compute loss with fake images
        z = torch.randn(BATCH_SIZE, 64).to(DEVICE)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, EPOCHS, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
        if (epoch+1) % 10 == 0 and (i+1) % 100 == 0 :
            fake_images = np.reshape(fake_images.data.numpy()[0],(28, 28))
            plt.imshow(fake_images, cmap = 'gray')
            plt.show()


# ## 참고
# 본 튜토리얼은 다음 자료를 참고하여 만들어졌습니다.
# * [yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) - MIT License
