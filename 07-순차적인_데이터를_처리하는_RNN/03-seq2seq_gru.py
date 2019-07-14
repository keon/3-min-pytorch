#!/usr/bin/env python
# coding: utf-8

# # Seq2Seq 기계 번역
# 이번 프로젝트에선 임의로 Seq2Seq 모델을 아주 간단화 시켰습니다.
# 한 언어로 된 문장을 다른 언어로 된 문장으로 번역하는 덩치가 큰 모델이 아닌
# 영어 알파벳 문자열("hello")을 스페인어 알파벳 문자열("hola")로 번역하는 Mini Seq2Seq 모델을 같이 구현해 보겠습니다.

import numpy as np
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim


vocab_size = 256  # 총 아스키 코드 개수
x_ = list(map(ord, "hello"))  # 아스키 코드 리스트로 변환
y_ = list(map(ord, "hola"))   # 아스키 코드 리스트로 변환
print("hello -> ", x_)
print("hola  -> ", y_)


x = torch.LongTensor(x_)
y = torch.LongTensor(y_)


'''
미니 GRU 모델.
'''
class Seq2Seq_GRU(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Seq2Seq_GRU, self).__init__()

        self.n_layers = 1
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size)
        self.decoder = nn.GRU(hidden_size * 2, hidden_size)
        self.project = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, targets):
        # Encoder inputs and states
        initial_state = self._init_state()
        embedding = self.embedding(inputs).unsqueeze(1)
        encoder_output, encoder_state = self.encoder(embedding, initial_state)
        outputs = []

        decoder_state = encoder_state
        for i in range(targets.size()[0]): 
            decoder_input = self.embedding(targets)[i].view(1,-1, self.hidden_size)
            decoder_input = th.cat((decoder_input, encoder_state), 2)
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            projection = self.project(decoder_output)#.unsqueeze(0))
            outputs.append(projection)
            
            #_, top_i = prediction.data.topk(1)
            
        outputs = th.stack(outputs, 1).squeeze()

        return outputs
    
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.n_layers, batch_size, self.hidden_size).zero_()) 


model = Seq2Seq_GRU(vocab_size, 16)
pred = model(x, y)


criterion = nn.CrossEntropyLoss()
optimizer = th.optim.Adam(model.parameters(), lr=1e-3)


y_.append(3)
y_label = Variable(th.LongTensor(y_[1:]))


print(y_label.shape)
print(y_label)


log = []
for i in range(10000):
    prediction = model(x, y)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_val = loss.data
    log.append(loss_val)
    if i % 100 == 0:
        print("%d loss: %s" % (i, loss_val.item()))
        _, top1 = prediction.data.topk(1, 1)
        for c in top1.squeeze().numpy().tolist():
            print(chr(c), end=" ")
        print()


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(log)
plt.xlim(0,150)
plt.ylim(0,15)
plt.ylabel('cross entropy loss')
plt.show()




