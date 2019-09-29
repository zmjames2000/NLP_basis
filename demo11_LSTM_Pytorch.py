# #-*- coding:utf-8 -*-

import torchtext
from torchtext.vocab import Vectors
import torch
import numpy as np
import random,os

USE_CUDA = torch.cuda.is_available()

random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
if USE_CUDA:
    torch.cuda.manual_seed_all(1000)

device = torch.device('cuda' if USE_CUDA else 'cpu')
BATCH_SIZE =  32
EMBEDDING_SIZE = 650
MAX_VOCAB_SIZE = 50000

DATA_PATH  = r'./data/demo10_pytorch_skip-Gram'
TRAIN_DATA = 'text8.train.txt'
TEST_DATA  = 'text8.test.txt'
VALI_DATA  = 'text8.dev.txt'
SAVE_MODEL = DATA_PATH + os.sep + 'lossm.pth'

TEXT = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path=DATA_PATH,
                                                                     text_field=TEXT,
                                                                     train=TRAIN_DATA,
                                                                     validation=VALI_DATA,
                                                                     test=TEST_DATA)
TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
VOCAB_SIZE = len(TEXT.vocab)
# print(VOCAB_SIZE, TEXT.vocab.itos[:100])
# TEXT.vocab.itos is list  TEXT.vocab.stoi is dict

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(datasets=(train, val, test),
                                                                     batch_size=BATCH_SIZE,
                                                                     device=device,
                                                                     bptt_len=50, #往回传的长度， 自定义
                                                                     repeat=False, # 写完文件不会重复
                                                                     shuffle=True)
# it = iter(train_iter)
# batch = next(it)
# print(batch)
# [torchtext.data.batch.Batch of size 32]
# 	[.text]:[torch.cuda.LongTensor of size 50x32 (GPU 0)]
# 	[.target]:[torch.cuda.LongTensor of size 50x32 (GPU 0)]
# print( ' '.join([TEXT.vocab.itos[i] for i in batch.text[:,0].data.cpu()]))
# anarchism originated as a term of abuse first used against early working class radicals including the diggers of the english revolution and the sans <unk> of the french revolution whilst the term is still used in a pejorative way to describe any act that used violent means to destroy the
# print( ' '.join([TEXT.vocab.itos[i] for i in batch.target[:,0].data.cpu()]))
# originated as a term of abuse first used against early working class radicals including the diggers of the english revolution and the sans <unk> of the french revolution whilst the term is still used in a pejorative way to describe any act that used violent means to destroy the organization
# 他们差了一个单词

class RNNModel(torch.nn.Module):
    def __init__(self, rnn_type, vocab_size, embed_size, hidden_size, nlayers, dropout=0.5):
        ''' 该模型包含以下几层:
            - 词嵌入层
            - 一个循环神经网络层(RNN, LSTM, GRU)
            - 一个线性层，从hidden state到输出单词表
            - 一个dropout层，用来做regularization
        '''
        super(RNNModel, self).__init__()
        self.drop = torch.nn.Dropout(dropout)
        # nn.Embedding -- rnn -- nn.Linear
        self.encoder = torch.nn.Embedding(vocab_size, embed_size) # 50000  650
        if rnn_type in ['LSTM','GRU']:
            self.rnn = getattr(torch.nn, rnn_type)(embed_size, hidden_size, nlayers, dropout=dropout)  # 650  hidden_size
        else:
            try:
                nonlinearity = {'RNN_TANH':'tanh', 'RNN_RELU':'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                        options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = torch.nn.RNN(embed_size, hidden_size, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.decoder = torch.nn.Linear(hidden_size, vocab_size) # hidden_size  vocab_size

        self.init_weights()

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, input, hidden): # 输入数据， 使用上面定义的层，输出数据
        ''' Forward pass:
            - word embedding
            - 输入循环神经网络
            - 一个线性层从hidden state转化为输出单词表
        '''
        # input.size  [ seg_length, batch_size]
        # nn.Embedding -- rnn -- nn.Linear
        #  self.encoder-- self.rnn--self.decoder
        emb = self.drop(self.encoder(input))  # [ seg_length, batch_size, embed_size]
        output,hidden = self.rnn(emb, hidden)
        # output : [ seg_length, batch_size, hidden_size]
        # hidden: [num_layer * batch_size * hidden_size, num_layer * batch_size * hidden_size ]
        # output = self.drop(output)
        decoded = self.decoder(self.drop(output.view(-1, output.size(2)))) # linear需要的是
        return decoded.view(output.size(0),output.size(1),-1), hidden

    def init_hidden(self, bsz, requires_grad=True):
        weight = next(self.parameters()) # 是iter
        if self.rnn_type =='LSTM':
            return (weight.new_zeros((self.nlayers, bsz, self.hidden_size), requires_grad=requires_grad),
                    weight.new_zeros((self.nlayers, bsz, self.hidden_size),requires_grad=requires_grad))
        else:
            return weight.new_zeros((self.nlayers, bsz, self.hidden_size), requires_grad=requires_grad)

model = RNNModel(rnn_type='LSTM',vocab_size=VOCAB_SIZE, embed_size=EMBEDDING_SIZE,hidden_size=100, nlayers=1)
if USE_CUDA:
    model = model.to(device)
# print(model)
# RNNModel(
#   (drop): Dropout(p=0.5, inplace=False)
#   (encoder): Embedding(50002, 650)
#   (rnn): LSTM(650, 100, dropout=0.5)
#   (decoder): Linear(in_features=100, out_features=50002, bias=True)
# )
# print(next(model.parameters()))

# 我们需要定义下面的一个function，帮助我们把一个hidden state和计算图之前的历史分离。
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)  # lstm 有2个  看  hidden = model.init_hidden(BATCH_SIZE)

# 我们首先定义评估模型的代码。
# 模型的评估和模型的训练逻辑基本相同，唯一的区别是我们只需要forward pass，不需要backward pass
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
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5) # 降learning_rate 0.5降一半
for epoch in range(NUM_EPOCHS):
    model.train() #  训练模式，测试模式，很多地方不一样，
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i, batch in enumerate(it):
        data, target = batch.text, batch.target
        # hidden status 可以一直往下传，但是内存，速度限制
        # 训练语言模型的时候，希望能按上面操作
        # 但是训练翻译的时候，就没有必要这么做了
        hidden = repackage_hidden(hidden)  # backprogate through all iter
        output,hidden = model(data, hidden) # 相当于执行 model.forward
        # output : [ seg_length, batch_size, hidden_size]

        loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1)) #   batch_size * target_class_dim , batch_size
        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if i%100 == 0:
            print('epoch',epoch,i,loss.item())

        if i%1000 == 0:
            val_loss = evaluate(model, val_iter)  # 模型评估，用验证集数据
            if len(val_losses) == 0 or val_loss < min(val_losses):
                torch.save(model.state_dict(), SAVE_MODEL) # 保存模型
                print('best model saved to lm.pth')
            else:
                # 模型的loss没有降下来，learning_rate decay
                scheduler.step()
                # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            val_losses.append(val_loss)