#!/usr/bin/env python
# coding: utf-8

# # Conditional GAN으로 생성 컨트롤하기

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
EPOCHS = 300
BATCH_SIZE = 100
USE_CUDA = torch.cuda.is_available()
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


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels]


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
        nn.Linear(64 + 10, 256),
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
    for i, (images, label) in enumerate(train_loader):
        images = images.reshape(BATCH_SIZE, -1).to(DEVICE)
        
        real_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE)
        fake_labels = torch.zeros(BATCH_SIZE, 1).to(DEVICE)

        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        class_label = one_hot_embedding(label, 10).to(DEVICE)
        z = torch.randn(BATCH_SIZE, 64).to(DEVICE)
        
        generator_input = torch.cat([z, class_label], 1)
        
        fake_images= G(generator_input)

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
        fake_images = G(generator_input)
        outputs = D(fake_images)
        
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch,
                          EPOCHS,
                          i+1,
                          total_step,
                          d_loss.item(),
                          g_loss.item(), 
                          real_score.mean().item(),
                          fake_score.mean().item()))


for i in range(100):
    label = torch.tensor([4])
    class_label = one_hot_embedding(label, 10).to(DEVICE)
    z = torch.randn(1, 64).to(DEVICE)
    generator_input = torch.cat([z, class_label], 1)
    fake_images= G(generator_input)
    fake_images = np.reshape(fake_images.cpu().data.numpy()[0],(28, 28))
    plt.imshow(fake_images, cmap = 'gray')
    plt.show()




