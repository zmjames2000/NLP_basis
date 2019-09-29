#-*- coding:utf-8 -*-

import torch
import torchtext
from torchtext.vocab import Vectors
import numpy as np
import random,os,time

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
if USE_CUDA:
    torch.cuda.manual_seed_all(1000)

DATA_PATH  = r'./data/demo10_pytorch_skip-Gram'
TRAIN_DATA = 'text8.train.txt'
TEST_DATA  = 'text8.test.txt'
VALI_DATA  = 'text8.dev.txt'
SAVE_MODEL = DATA_PATH + os.sep + 'loss_model2.th'

BATCH_SIZE =  32
EMBEDDING_SIZE = 650
MAX_VOCAB_SIZE = 50000

TEXT = torchtext.data.Field(lower=True)
train,val,test = torchtext.datasets.LanguageModelingDataset.splits(path=DATA_PATH,
                                                                   text_field=TEXT,
                                                                   train=TRAIN_DATA,
                                                                   validation=VALI_DATA,
                                                                   test=TEST_DATA)
TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE) # 构建语料库
train_iter,val_iter,test_iter = torchtext.data.BPTTIterator.splits(datasets=(train,val,test),
                                                                   batch_size=BATCH_SIZE,
                                                                   device=device,
                                                                   bptt_len=50, # 往回传的长度，
                                                                   shuffle=True)

VOCAB_SIZE = len(TEXT.vocab)

class RNNModel(torch.nn.Module):
    def __init__(self, rnn_type, vocab_size, embed_size, hidden_size, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = torch.nn.Dropout(dropout)
        # nn.Embedding -- rnn -- nn.Linear
        self.encoder = torch.nn.Embedding(vocab_size,embed_size)
        if rnn_type in ['LSTM',"GRU"]:
            self.rnn = getattr(torch.nn, rnn_type)(embed_size,hidden_size,nlayers,dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                        options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = torch.nn.RNN(embed_size,hidden_size,nlayers,nonlinearity=nonlinearity,dropout=dropout)

        self.decorder = torch.nn.Linear(hidden_size,vocab_size)

        self.init_weights()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decorder.weight.data.uniform_(-initrange, initrange)
        self.decorder.bias.data.zero_()

    def forward(self, input_data, hidden):
        # nn.Embedding -- rnn -- nn.Linear
        #  self.encoder-- self.rnn--self.decoder
        emb = self.drop(self.encoder(input_data))
        output,hidden = self.rnn(emb, hidden)
        decoded = self.decorder(self.drop(output.view(-1, output.size(2))))
        return  decoded.view(output.size(0),output.size(1),-1), hidden

    def init_hidden(self, bsz, requires_grad=True):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((self.nlayers,bsz,self.hidden_size), requires_grad=requires_grad),
                    weight.new_zeros((self.nlayers,bsz,self.hidden_size), requires_grad=requires_grad))
        else:
            return  weight.new_zeros((self.nlayers, bsz, self.hidden_size), requires_grad=requires_grad)

model = RNNModel(rnn_type='LSTM', vocab_size=VOCAB_SIZE,embed_size=EMBEDDING_SIZE,hidden_size=100,nlayers=2)
if USE_CUDA:
    model = model.to(device)

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple( repackage_hidden(v) for v in h) # LSTM 有两个

def evaluate(model, data):# 处理验证集的数据
    model.eval()
    total_loss = 0.
    it = iter(data)
    total_count = 0
    with torch.no_grad(): # 确保下面没有 grad
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            total_count += np.multiply(*data.size())
            total_loss  += loss.item() * np.multiply(*data.size()) # *data.size()相当于把元素拆开，相乘

    loss = total_loss/total_count
    model.train()
    return  loss

NUM_EPOCHS = 2
GRAD_CLIP = 5.
val_losses = []
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.002
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5) # half
for epoch in range(NUM_EPOCHS):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    start_time = time.time()
    for i, batch in enumerate(it):
        data, target =  batch.text, batch.target
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1)) # view source

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP) #梯度裁剪原理 设定阈值，当梯度小于/大于阈值时，更新的梯度为阈值
        optimizer.step()

        if i%100 == 0:
            print('epoch', epoch, i, loss.item())

        if i%1000 == 0:
            end_time = time.time()
            print('time:{}'.format(end_time - start_time))

            val_loss = evaluate(model, val_iter)
            if len(val_losses) == 0 or val_loss < min(val_losses):
                torch.save(model.state_dict(), SAVE_MODEL)
                print('best model saved to {}'.format(SAVE_MODEL))
            elif i%5000 == 0:
                scheduler.step()
            val_losses.append(val_loss)

