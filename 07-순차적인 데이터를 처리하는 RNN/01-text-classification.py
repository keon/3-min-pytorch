#!/usr/bin/env python
# coding: utf-8

# # 프로젝트 1. 영화 리뷰 감정 분석
# **RNN 을 이용해 IMDB 데이터를 가지고 텍스트 감정분석을 해 봅시다.**
# 데이터의 순서정보를 학습한다는 점에서 RNN은 CIFAR10 같이 정적인 고정된 형태의 데이터 보다는 동영상, 자연어, 주가 변동 데이터 등의 동적인 시계열 데이터를 이용할때 퍼포먼스가 극대화됩니다.
# 이번 프로젝트를 통해 가장 기본적인 자연어 처리(Natural Language Processing)작업이라고 할 수 있는 '텍스트 감정분석'(Text Sentiment Analysis)모델을 RNN을 이용해 구현하고 학습시켜 보겠습니다.
# 이번 책에서 처음으로 접하는 텍스트 형태의 데이터셋인 IMDB 데이터셋은 50,000건의 영화 리뷰로 이루어져 있습니다.
# 각 리뷰는 다수의 영어 문장들로 이루어져 있으며, 평점이 7점 이상의 긍정적인 영화 리뷰는 2로, 평점이 4점 이하인 부정적인 영화 리뷰는 1로 레이블링 되어 있습니다. 영화 리뷰 텍스트를 RNN 에 입력시켜 영화평의 전체 내용을 압축하고, 이렇게 압축된 리뷰가 긍정적인지 부정적인지 판단해주는 간단한 분류 모델을 만드는 것이 이번 프로젝트의 목표입니다.

# ### 워드 임베딩
# 본격적으로 모델에 대한 코드를 짜보기 전, 자연어나 텍스트 데이터를 가지고 딥러닝을 할때 언제나 사용되는 ***워드 임베딩(Word Embedding)***에 대해 간단히 배워보겠습니다.
# IMDB 데이터셋은 숫자로 이뤄진 텐서나 벡터 형태가 아닌 순전한 자연어로 이뤄져 있습니다. 이러한 데이터를 인공신경망에 입력시키기 위해선 여러 사전처리를 해 데이터를 벡터로 나타내 줘야 합니다. 이를 위해 가장 먼저 해야 할 일은 아래와 같이 영화 리뷰들을 단어 단위의 토큰으로 나누어 주는 것입니다. 간단한 데이터셋에선 파이썬의 split(‘ ’) 함수를 써서 토크나징을 해 줘도 큰 문제는 없지만, 더 깔끔한 토크나이징을 위해 Spacy 같은 오픈소스를 사용하는걸 추천드립니다.
# ```python
# ‘It was a good movie.’ → [‘it’, ‘was’, ‘a’, ‘good’, ‘movie’]
# ```
# 그 후 영화평 속의 모든 단어는 one hot encoding 이라는 기법을 이용해 벡터로 변환됩니다. 예를 들어 데이터셋에 총 10개의 다른 단어들이 있고 ‘movie’ 라는 단어가 10개의 단어 중 3번째 단어 일 경우 'movie'는 다음과 같이 나타내어 집니다.
# ```python
# movie = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# ```
# 그 다음으로는 one hot encoding을 거친 단어 벡터를 '사전 속 낱말 수' X '임베딩 차원값' 모양의 랜덤한 임베딩 행렬(Embedding Matrix)과 행렬곱을 해 주어야 합니다. 행렬곱의 결과는 'movie'라는 단어를 대표하는 다양한 특성값을 가진 벡터입니다.
# 워드 임베딩은 언뜻 보기에도 코드로 정의하기엔 골치아픈 동작이지만,
# 다행히도 파이토치의 nn.Embedding() 함수를 사용하면 별로 어렵지 않게 이뤄낼 수 있습니다.

# 그럼 본격적으로 코드를 짜 보겠습니다.  
# 가장 먼저 모델 구현과 학습에 필요한 라이브러리 들을 임포트 해 줍니다.
# ```python
# import os
# import sys
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchtext import data, datasets
# ```

# 모델 구현과 학습에 필요한 하이퍼파라미터 들을 정의해 줍니다.
# ```python
# BATCH_SIZE = 64
# lr = 0.001
# EPOCHS = 40
# torch.manual_seed(42)
# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
# ```

# BasicLSTM 라고 하는 RNN을 포함하는 신경망 모델을 만들어 보겠습니다. 여타 다른 신경망 모델과 같이 파이토치의 nn.Module 을 상속받습니다.
# ```python
# class BasicLSTM(nn.Module):
#     def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
#         super(BasicLSTM, self).__init__()
#         print("Building Basic LSTM model...")
# ```
# __init()__ 함수 속 가장 먼저 정의되는 변수는 RNN의 '층'이라고 할 수 있는 n_layers 입니다. 아주 복잡한 모델이 아닌 이상, n_layers는 2이하의 값으로 정의되는게 보통입니다.  
# 앞에서 잠시 언급했던 nn.Embedding() 함수는 2개의 파라미터를 입력받습니다. 이 중 첫번째는 전체 데이터셋 속 모든 단어를 사전 형태로 나타냈을 때
# 이 사전속 단어의 갯수라고 할 수 있는 n_vocab이라는 숫자입니다.
# 두번째 파라미터는 embed 라는 숫자입니다. 이 숫자는 쉽게 말해 임베딩된 단어 텐서가 지니는 차원값 이라고 할 수 있습니다.
# 즉, 한 영화 리뷰속 모든 단어가 임베딩을 거치면 영화 리뷰는 embed 만큼 특성값을 지닌 단어 텐서들이 차례대로 나열된 배열 형태로 나타내어 집니다.
# ```python
#         self.n_layers = n_layers
#         self.embed = nn.Embedding(n_vocab, embed_dim)
# ```
# 다음으로 drop out 을 정의해 주고 본격적으로 RNN 모델을 정의합니다. 사실 원시적인 RNN은 입력받은 시계열 데이터의 길이가 길어지면 
# 학습 도중 경사값(Gradient)이 너무 작아져 버리거나 너무 커져 버리는 고질적인 문제가 있었습니다.
# 따라서 이러한 문제에서 조금 더 자유로운 LSTM(Long Short Term Memory)라는 RNN 을 사용하겠습니다.
# ```python
#         self.lstm = nn.LSTM(embed_dim, self.hidden_dim,
#                             num_layers=self.n_layers,
#                             dropout=dropout_p,
#                             batch_first=True)
# ```
# 앞에서 설명했듯, RNN은 텐서의 배열을 하나의 텐서로 압축시킵니다.
# 하지만 사실상 RNN의 기능은 여기서 끝나기 때문에 모델이 영화 리뷰가 긍정적인지 부정적인지 분류하는 결과값을 출력하려면 압축된 텐서를 다음과 같이 다층신경망에 입력시켜야 합니다.
# ```python
#         self.out = nn.Linear(self.hidden_dim, n_classes)
# ```
# 본격적으로 모델에 입력된 텍스트 데이터가 어떤 전처리 과정을 거치고 신경망에 입력되는지 정의하는 forward 함수를 구현합니다.
# 모델에 입력되는 데이터 x 는 한 batch 속에 있는 모든 영화평 입니다.
# 이들이 embed 함수를 통해 워드 임베딩을 하게 되면 LSTM 에 입력될 수 있는 형태가 됩니다.
# ```python
#     def forward(self, x):
#         x = self.embed(x)  #  [b, i] -> [b, i, e]
# ```
# 보통의 신경망이라면 이제 바로 신경망 모듈의 forward 함수를 호출해도 되겠지만 
# LSTM과 같은 RNN 계열의 신경망은 입력 데이터 말고도 밑의 코드처럼 은닉 벡터(Hidden Vector)라는 텐서를 정의하고 신경망에 입력해 줘야 합니다.
# ```python
#         h_0 = self._init_state(batch_size=x.size(0))
#         x, _ = self.lstm(x, h_0)  # [i, b, h]
# ```
# 첫번째 은닉 벡터(Hidden Vector) 인 h_0을 생성하는 _init_state 함수를 구현합니다. 꼭 그럴 필요는 없으나, 첫번째 은닉 벡터는 아래의 코드처럼 모든 특성값이 0인 벡터로 설정해 주는 것이 보통입니다.
# ```python
#     def _init_state(self, batch_size=1):
#         weight = next(self.parameters()).data
#         return (
#             weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
#             weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
#         )
# ```
# next(self.parameters()).data 를 통해 모델 속 가중치 텐서를 weight 이라는 변수에 대입시킵니다.
# 그리고 new() 함수를 이용해 weight 텐서와 같은 자료형을 갖고 있지만 (n_layers, batch_size, hidden_dim)꼴의 텐서 두개를 정의합니다. 그리고 이 두 텐서에 zero_() 함수를 호출함으로써 텐서 속 모든 원소값을 0으로 바꿔줍니다. 대부분의 RNN 계열의 신경망은 은닉 벡터를 하나만을 요구하지만, 좀 더 복잡한 구조를 가진 LSTM 은 이렇게 같은 모양의 텐서 두 개를 정의해 줘야 합니다.
# RNN이 만들어낸 마지막 은닉 벡터를 h_t 라고 정의하겠습니다.
# ```python
#         h_t = x[:,-1,:]
# ```
# 이제 영화 리뷰속 모든 내용을 압축한 h_t를 다층신경망에 입력시켜 결과를 출력해야 합니다.
# ```python
#         logit = self.out(h_t)  # [b, h] -> [b, o]
#         return logit
# ```
# 모델 구현과 신경망 학습에 필요한 함수를 구현했으면 본격적으로 IMDB 데이터셋을 가져와 보겠습니다.
# 사실 아무 가공처리를 가하지 않은 텍스트 형태의 데이터셋을 신경망에 입력하는데까지는 매우 번거로운 작업을 필요로합니다.
# 그러므로 우리는 이러한 전처리 작업들을 대신 해주는 Torch Text라이브러리를 사용해 IMDB 데이터셋을 가져오겠습니다.
# 가장 먼저 텍스트 형태의 영화 리뷰들과 그에 해당하는 레이블을 텐서로 바꿔줄 때 필요한 설정사항들을 정해줘야 합니다.
# 그러기 위해 이러한 설정정보를 담고있는 TEXT 와 LABEL 이라는 객체를 생성합니다. 
# ```python
# TEXT = data.Field(sequential=True, batch_first=True, lower=True)
# LABEL = data.Field(sequential=False, batch_first=True)
# ```
# sequential 이라는 파라미터를 이용해 데이터셋이 순차적 데이터셋이라고 명시해 주고 batch_first 파라미터로 신경망에 입력되는 텐서의 첫번째 차원값이 batch_size 가 되도록 정해줍니다.
# 마지막으로 lower 변수를 이용해 텍스트 데이터 속 모든 영문 알파벳이 소문자가 되도록 설정해 줍니다.
# 그 다음으로는 datasets 객체의 splits 함수를 이용해 모델에 입력되는 데이터셋을 만들어줍니다.
# ```python
# train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
# ```
# 이제 만들어진 데이터셋을 이용해 전에 설명한 워드 임베딩에 필요한 워드 사전(Word Vocabulary)를 만들어줍니다.
# ```python
# TEXT.build_vocab(train_data, min_freq=5)
# LABEL.build_vocab(train_data)
# ```
# min_freq 은 학습데이터 속에서 최소한 5번 이상 등장한 단어들만을 사전속에 정의하겠다는 뜻입니다. 즉 학습 데이터 속에서 드물게 출현하는 단어는 'unk'(Unknown) 이라는 토큰으로 정의됩니다.
# 그 다음으로는 train_data 와 test_data 에서 batch tensor 를 generate 할 수 있는 iterator 를 만들어 줍니다.
# ```python
# train_iter, test_iter = data.BucketIterator.splits(
#         (train_data, test_data), batch_size=BATCH_SIZE,
#         shuffle=True, repeat=False)
# ```
# 마지막으로 사전 속 단어들의 숫자와 레이블의 수를 정해주는 변수를 만들어 줍니다.
# ```python
# vocab_size = len(TEXT.vocab)
# n_classes = 2
# ```

# 그 다음은 train() 함수와 evaluate() 함수를 구현할 차례입니다.
# ```python
# def train(model, optimizer, train_iter):
#     model.train()
#     for b, batch in enumerate(train_iter):
#         x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
#         y.data.sub_(1)  # index align
#         optimizer.zero_grad()
#         logit = model(x)
#         loss = F.cross_entropy(logit, y)
#         loss.backward()
#         optimizer.step()
#         if b % 100 == 0:
#             corrects = (logit.max(1)[1].view(y.size()).data == y.data).sum()
#             accuracy = 100.0 * corrects / batch.batch_size
#             sys.stdout.write(
#                 '\rBatch[%d] - loss: %.6f  acc: %.2f' %
#                 (b, loss.item(), accuracy))
# ```
# ```python
# def evaluate(model, val_iter):
#     """evaluate model"""
#     model.eval()
#     corrects, avg_loss = 0, 0
#     for batch in val_iter:
#         x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
# #         x, y = batch.text, batch.label
#         y.data.sub_(1)  # index align
#         logit = model(x)
#         loss = F.cross_entropy(logit, y, size_average=False)
#         avg_loss += loss.item()
#         corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
#     size = len(val_iter.dataset)
#     avg_loss = avg_loss / size
#     accuracy = 100.0 * corrects / size
#     return avg_loss, accuracy
# ```
# 본격적으로 학습을 시작하기 전, 모델 객체와 최적화 알고리즘을 정의합니다.
# ```python
# model = BasicLSTM(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# ```
# 이제 학습에 필요한 모든 준비는 되었습니다. 마지막으로 학습을 하는 loop을 구현합니다.
# ```python
# best_val_loss = None
# for e in range(1, EPOCHS+1):
#     train(model, optimizer, train_iter)
#     val_loss, val_accuracy = evaluate(model, test_iter)
#     print("\n[Epoch: %d] val_loss:%5.2f | acc:%5.2f" % (e, val_loss, val_accuracy))
# ``` 
# 4장에서 배워 봤듯이, 우리가 원하는 최종 모델은 Training Loss가 아닌 Validation Loss가 최소화된 모델입니다. 다음과 같이 Validation Loss가 가장 작은 모델을 저장하는 로직을 구현합니다.
# ```python    
#         # Save the model if the validation loss is the best we've seen so far.
#     if not best_val_loss or val_loss < best_val_loss:
#         if not os.path.isdir("snapshot"):
#             os.makedirs("snapshot")
#         torch.save(model.state_dict(), './snapshot/convcnn.pt')
#         best_val_loss = val_loss
# ```

# ### 전체 코드
# ```python
# import os
# import sys
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchtext import data, datasets
# BATCH_SIZE = 64
# lr = 0.001
# EPOCHS = 40
# torch.manual_seed(42)
# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
# class BasicLSTM(nn.Module):
#     def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
#         super(BasicLSTM, self).__init__()
#         print("Building Basic LSTM model...")
#         self.n_layers = n_layers
#         self.embed = nn.Embedding(n_vocab, embed_dim)
#         self.hidden_dim = hidden_dim
#         self.dropout = nn.Dropout(dropout_p)
#         self.lstm = nn.LSTM(embed_dim, self.hidden_dim,
#                             num_layers=self.n_layers,
#                             dropout=dropout_p,
#                             batch_first=True)
#         self.out = nn.Linear(self.hidden_dim, n_classes)
#     def forward(self, x):
#         x = self.embed(x)  #  [b, i] -> [b, i, e]
#         h_0 = self._init_state(batch_size=x.size(0))
#         x, _ = self.lstm(x, h_0)  # [i, b, h]
#         h_t = x[:,-1,:]
#         self.dropout(h_t)
#         logit = self.out(h_t)  # [b, h] -> [b, o]
#         return logit
#     def _init_state(self, batch_size=1):
#         weight = next(self.parameters()).data
#         return (
#             weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
#             weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
#         )
# print("\nLoading data...")
# TEXT = data.Field(sequential=True, batch_first=True, lower=True)
# LABEL = data.Field(sequential=False, batch_first=True)
# train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
# TEXT.build_vocab(train_data, min_freq=5)
# LABEL.build_vocab(train_data)
# train_iter, test_iter = data.BucketIterator.splits(
#         (train_data, test_data), batch_size=BATCH_SIZE,
#         shuffle=True, repeat=False)
# vocab_size = len(TEXT.vocab)
# n_classes = 2  
# def train(model, optimizer, train_iter):
#     model.train()
#     for b, batch in enumerate(train_iter):
#         x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
# #         x, y = batch.text, batch.label
#         y.data.sub_(1)  # index align
#         optimizer.zero_grad()
#         logit = model(x)
#         loss = F.cross_entropy(logit, y)
#         loss.backward()
#         optimizer.step()
#         if b % 100 == 0:
#             corrects = (logit.max(1)[1].view(y.size()).data == y.data).sum()
#             accuracy = 100.0 * corrects / batch.batch_size
#             sys.stdout.write(
#                 '\rBatch[%d] - loss: %.6f  acc: %.2f' %
#                 (b, loss.item(), accuracy))
# def evaluate(model, val_iter):
#     """evaluate model"""
#     model.eval()
#     corrects, avg_loss = 0, 0
#     for batch in val_iter:
#         x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
# #         x, y = batch.text, batch.label
#         y.data.sub_(1)  # index align
#         logit = model(x)
#         loss = F.cross_entropy(logit, y, size_average=False)
#         avg_loss += loss.item()
#         corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
#     size = len(val_iter.dataset)
#     avg_loss = avg_loss / size
#     accuracy = 100.0 * corrects / size
#     return avg_loss, accuracy
# print("[TRAIN]: %d \t [TEST]: %d \t [VOCAB] %d \t [CLASSES] %d"
#       % (len(train_iter),len(test_iter), vocab_size, n_classes))
# model = BasicLSTM(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# print(model)
# best_val_loss = None
# for e in range(1, EPOCHS+1):
#     train(model, optimizer, train_iter)
#     val_loss, val_accuracy = evaluate(model, test_iter)
#     print("\n[Epoch: %d] val_loss:%5.2f | acc:%5.2f" % (e, val_loss, val_accuracy))
#     # Save the model if the validation loss is the best we've seen so far.
#     if not best_val_loss or val_loss < best_val_loss:
#         if not os.path.isdir("snapshot"):
#             os.makedirs("snapshot")
#         torch.save(model.state_dict(), './snapshot/convcnn.pt')
#         best_val_loss = val_loss
# ```

# ## 원본 코드

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets


# get hyper parameters
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 40
torch.manual_seed(42)
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


# load data
print("\nLoading data...")
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
TEXT.build_vocab(train_data, min_freq=5)
LABEL.build_vocab(train_data)

# train_iter, test_iter = data.BucketIterator.splits(
#         (train_data, test_data), batch_size=BATCH_SIZE,
#         shuffle=True, repeat=False,device=-1)
train_iter, test_iter = data.BucketIterator.splits(
        (train_data, test_data), batch_size=BATCH_SIZE,
        shuffle=True, repeat=False)


vocab_size = len(TEXT.vocab)
n_classes = 2
#len(LABEL.vocab) - 1


print("[TRAIN]: %d \t [TEST]: %d \t [VOCAB] %d \t [CLASSES] %d"
      % (len(train_iter),len(test_iter), vocab_size, n_classes))








class BasicGRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(BasicGRU, self).__init__()
        print("Building Basic GRU model...")
        self.n_layers = n_layers
        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                            num_layers=self.n_layers,
                            dropout=dropout_p,
                            batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0))
        x, _ = self.gru(x, h_0)  # [i, b, h]
        h_t = x[:,-1,:]
        self.dropout(h_t)
        logit = self.out(h_t)  # [b, h] -> [b, o]
        return logit
    
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()


def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
#         x, y = batch.text, batch.label
        y.data.sub_(1)  # index align
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()
        if b % 100 == 0:
            corrects = (logit.max(1)[1].view(y.size()).data == y.data).sum()
            accuracy = 100.0 * corrects / batch.batch_size
            sys.stdout.write(
                '\rBatch[%d] - loss: %.6f  acc: %.2f' %
                (b, loss.item(), accuracy))


def evaluate(model, val_iter):
    """evaluate model"""
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
#         x, y = batch.text, batch.label
        y.data.sub_(1)  # index align
        logit = model(x)
        loss = F.cross_entropy(logit, y, size_average=False)
        avg_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = avg_loss / size
    accuracy = 100.0 * corrects / size
    return avg_loss, accuracy


model = BasicGRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print(model)


best_val_loss = None
for e in range(1, EPOCHS+1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, test_iter)

    print("\n[Epoch: %d] val_loss:%5.2f | acc:%5.2f" % (e, val_loss, val_accuracy))
    
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss


# class BasicRNN(nn.Module):
#     """
#         Basic RNN
#     """
#     def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
#         super(BasicRNN, self).__init__()
#         print("Building Basic RNN model...")
#         self.n_layers = n_layers
#         self.hidden_dim = hidden_dim

#         self.embed = nn.Embedding(n_vocab, embed_dim)
#         self.dropout = nn.Dropout(dropout_p)
#         self.rnn = nn.RNN(embed_dim, hidden_dim, n_layers,
#                           dropout=dropout_p, batch_first=True)
#         self.out = nn.Linear(self.hidden_dim, n_classes)

#     def forward(self, x):
#         embedded = self.embed(x)  #  [b, i] -> [b, i, e]
#         _, hidden = self.rnn(embedded)
#         self.dropout(hidden)
#         hidden = hidden.squeeze()
#         logit = self.out(hidden)  # [b, h] -> [b, o]
#         return logit

