
# coding: utf-8

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

class BasicLSTM(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(BasicLSTM, self).__init__()
        print("Building Basic LSTM model...")
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embed_dim, self.hidden_dim,
                            num_layers=self.n_layers,
                            dropout=dropout_p,
                            batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)  #  [b, i] -> [b, i, e]
        h_0 = self._init_state(batch_size=x.size(0))
        x, _ = self.lstm(x, h_0)  # [i, b, h]
        h_t = x[:,-1,:]
        self.dropout(h_t)
        logit = self.out(h_t)  # [b, h] -> [b, o]
        return logit
    
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        )


def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)  # index align
        optimizer.zero_grad()
        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()
#         if b % 100 == 0:
#             corrects = (logit.max(1)[1].view(y.size()).data == y.data).sum()
#             accuracy = 100.0 * corrects / batch.batch_size
#             sys.stdout.write(
#                 '\rBatch[%d] - loss: %.6f  acc: %.2f' %
#                 (b, loss.item(), accuracy))


def evaluate(model, val_iter):
    """evaluate model"""
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)  # index align
        logit = model(x)
        loss = F.cross_entropy(logit, y, size_average=False)
        avg_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = avg_loss / size
    accuracy = 100.0 * corrects / size
    return avg_loss, accuracy


# # IMDB 데이터셋 가져오기

# load data
print("\nLoading data...")
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
TEXT.build_vocab(train_data, min_freq=5)
LABEL.build_vocab(train_data)

train_iter, test_iter = data.BucketIterator.splits(
        (train_data, test_data), batch_size=BATCH_SIZE,
        shuffle=True, repeat=False)

vocab_size = len(TEXT.vocab)
n_classes = len(LABEL.vocab) - 1


print("[TRAIN]: %d \t [TEST]: %d \t [VOCAB] %d \t [CLASSES] %d"
      % (len(train_iter),len(test_iter), vocab_size, n_classes))


model = BasicLSTM(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print(model)


best_val_loss = None
for e in range(1, EPOCHS+1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, test_iter)

    print("\n[Epoch: %d] val_loss:%5.2f | acc:%5.2f" % (e, val_loss, val_accuracy))
    
    # Save the model if the validation loss is the best we've seen so far.
#     if not best_val_loss or val_loss < best_val_loss:
#         if not os.path.isdir("snapshot"):
#             os.makedirs("snapshot")
#         torch.save(model.state_dict(), './snapshot/convcnn.pt')
#         best_val_loss = val_loss

