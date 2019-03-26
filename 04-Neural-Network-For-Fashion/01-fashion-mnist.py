#!/usr/bin/env python
# coding: utf-8

# # 4.1 Fashion MNIST 데이터셋 알아보기

get_ipython().run_line_magic('matplotlib', 'inline')
from torchvision import datasets, transforms, utils
from torch.utils import data

import matplotlib.pyplot as plt
import numpy as np


# ## [개념] Fashion MNIST 데이터셋

transform = transforms.Compose([
    transforms.ToTensor()
])


trainset = datasets.FashionMNIST(
    root      = './.data/', 
    train     = True,
    download  = True,
    transform = transform
)
testset = datasets.FashionMNIST(
    root      = './.data/', 
    train     = False,
    download  = True,
    transform = transform
)


batch_size = 16

train_loader = data.DataLoader(
    dataset     = trainset,
    batch_size  = batch_size
)
test_loader = data.DataLoader(
    dataset     = testset,
    batch_size  = batch_size
)


dataiter       = iter(train_loader)
images, labels = next(dataiter)


# ## 멀리서 살펴보기
# 누군가 "숲을 먼저 보고 나무를 보라"고 했습니다. 데이터셋을 먼저 전체적으로 살펴보며 어떤 느낌인지 알아보겠습니다.

img   = utils.make_grid(images, padding=0)
npimg = img.numpy()
plt.figure(figsize=(10, 7))
plt.imshow(np.transpose(npimg, (1,2,0)))
plt.show()


CLASSES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

KR_CLASSES = {
    0: '티셔츠/윗옷',
    1: '바지',
    2: '스웨터',
    3: '드레스',
    4: '코트',
    5: '샌들',
    6: '셔츠',
    7: '운동화',
    8: '가방',
    9: '앵클부츠'
}

for label in labels:
    index = label.item()
    print(KR_CLASSES[index])


# ## 가까이서 살펴보기
# 또 누군가는 "숲만 보지 말고 나무를 보라"고 합니다. 이제 전체적인 느낌을 알았으니 개별적으로 살펴보겠습니다.

idx = 1

item_img = images[idx]
item_npimg = item_img.squeeze().numpy()
plt.title(CLASSES[labels[idx].item()])
print(item_npimg.shape)
plt.imshow(item_npimg, cmap='gray')
plt.show()


# plot one example
print(trainset.train_data.size())     # (60000, 28, 28)
print(trainset.train_labels.size())   # (60000)
plt.imshow(trainset.train_data[1].numpy(), cmap='gray')
plt.title('%i' % trainset.train_labels[1])
plt.show()


img.max()


img.min()


img

