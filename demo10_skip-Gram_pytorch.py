# #-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random
import  math,os
from tqdm import  tqdm,trange

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

USE_CUDA = torch.cuda.is_available()
# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
if USE_CUDA:
    torch.cuda.manual_seed_all(1000)

# 设定一些超参数
K = 100  # number of negative samples
C = 3  # nearby words threshold
NUM_EPOCHS = 2  # The number of epochs of training
MAX_VOCAB_SIZE = 30000  # the vocabulary size
BATCH_SIZE = 128  # the batch size
LEARNING_RATE = 0.1  # the initial learning rate
EMBEDDING_SIZE = 100


# 从文本文件中读取所有的文字，通过这些文本创建一个vocabulary
# 由于单词数量可能太大，我们只选取最常见的MAX_VOCAB_SIZE个单词
# 我们添加一个UNK单词表示所有不常见的单词
# 我们需要记录单词到index的mapping，以及index到单词的mapping，单词的count，单词的(normalized) frequency，以及单词总数。
DATA_PATH = r'./data/demo10_pytorch_skip-Gram'
TRAIN_DATA = DATA_PATH + os.sep + 'text8.train.txt'
TEST_DATA  = DATA_PATH + os.sep + 'text8.test.txt'
with open(TRAIN_DATA,'r',encoding='utf-8') as fin:
    text = fin.read()

text = text.split()
text = [w for w in text]
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
vocab['<unk>'] = len(text) - np.sum(list(vocab.values()))  #{ 'the':8459854}
idx_to_word  = [word for word in vocab.keys()]
word_to_idx  = { word:key for key, word in vocab.items()}
word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
word_freqs  = word_counts / np.sum(word_counts)
word_freqs  = word_freqs ** (3./4.)
word_freqs  = word_freqs / np.sum(word_freqs)
VOCAB_SIZE  = len(idx_to_word)


# 一个dataloader需要以下内容：
#     把所有text编码成数字，然后用subsampling预处理这些文字。
#     保存vocabulary，单词count，normalized word frequency
#     每个iteration sample一个中心词
#     根据当前的中心词返回context单词
#     根据中心词sample一些negative单词
#     返回单词的counts
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(WordEmbeddingDataset, self).__init__()
        self.word_to_idx = word_to_idx # dict
        self.idx_to_word = idx_to_word # list
        self.word_freqs  = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)
        self.text_encoded = [word_to_idx.get(t, VOCAB_SIZE-1) for t in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx-C, idx)) + list(range(idx+1, idx+C+1))
        pos_indices =  [ i%len(self.text_encoded) for i in pos_indices]
        pos_words  = self.text_encoded[pos_indices]
        neg_words  = torch.multinomial(self.word_freqs, K * pos_words.shape[0], replacement=True)

        return center_word, pos_words, neg_words

# 有了dataloader之后，我们可以轻松随机打乱整个数据集，拿到一个batch的数据等等。
dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#BrokenPipeError: [Errno 32] Broken pipe
# print(next(iter(dataloader)))

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size,sparse=False) # 500000 * 100
        self.out_embed.weight.data.uniform_(-initrange, initrange)
        self.in_embed  = nn.Embedding(self.vocab_size, self.embed_size,sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_lables, neg_labes):
        batch_size = input_labels.size(0)
        input_embedding = self.in_embed(input_labels) # B * embed_size
        pos_embedding   = self.out_embed(pos_lables)  # B * (2*C) * embed_size
        neg_embedding   = self.out_embed(neg_labes)   # B * (2*C * K) * embed_size

        pos_dot = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze() # B * (2*C)
        neg_dot = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze() # B * (2*C*K)

        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)
        loss = log_pos + log_neg

        return  -loss

    def input_embedding(self):
        return self.in_embed.weight.data.cpu().numpy()

model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
if USE_CUDA:
    print('use cuda......')
    model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
for epoch in trange(NUM_EPOCHS):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        input_labels = input_labels.long()
        pos_labels   = pos_labels.long()
        neg_labels   = neg_labels.long()

        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels   = pos_labels.cuda()
            neg_labels   = neg_labels.cuda()

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if i%100 == 0:
            print("epoch", epoch, i, loss.item())


# while True: input('>>>>>ending')




