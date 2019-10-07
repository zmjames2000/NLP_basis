#-*- coding:utf-8 -*-

from grammer.logger import Logger
from grammer.timmer import epoch_time

import torch
import torch.nn.functional as F
import time,os,random

import torchtext
from torchtext import datasets

USE_CUDA = torch.cuda.is_available()
device = torch.device( 'cuda' if USE_CUDA  else 'cpu')
DIR_PATH  = r'./data/demo14'
SAVE_MODEL = DIR_PATH + os.sep + 'loss_model.pth'
SAVE_LOG   = DIR_PATH + os.sep + 'lstm-model.pt'

logger = Logger(SAVE_LOG)

SEED = 100
torch.random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

TEXT = torchtext.data.Field(tokenize='spacy')
LABEL= torchtext.data.LabelField(dtype=torch.float)
train_data,test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data,valid_data = train_data.split(random_state=random.seed(SEED))
logger.info(f'Number of training examples: {len(train_data)}')
logger.info(f'Number of validation examples: {len(valid_data)}')
logger.info(f'Number of testing examples: {len(test_data)}')
TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)
# print(TEXT.vocab.freqs.most_common(20))
# print(TEXT.vocab.itos[:10])
# print(LABEL.vocab.stoi)

BATCH_SIZE = 64
train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(datasets=(train_data,valid_data,test_data),
                                                                         batch_size=BATCH_SIZE,
                                                                         device=device)

class RNN(torch.nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_dim, ouput_dim,\
                 n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size,embedding_dim,padding_idx=pad_idx)
        self.rnn       = torch.nn.LSTM(embedding_dim,hidden_dim,num_layers=n_layers, \
                                       bidirectional=bidirectional,dropout=dropout)  # LSTM 效果比较好
        self.fc        = torch.nn.Linear(hidden_dim * 2, ouput_dim)
        self.dropout   = torch.nn.Dropout(dropout) # 用在embedding 后面

    def forward(self, text):
        embedded = self.dropout(self.embedding(text)) #[sent len, batch size, emb dim]
        output,(hidden, cell) = self.rnn(embedded)  # rnn(embedded, hidden) 可以不传，不传的话默认把全0的向量传入
        #output,是每一个位置传出的 hidden是最后一个位置传出的 hidden是我们想要的语言输出
        # output = [sent len, batch size, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # [batch size, hid dim * num directions]
        return self.fc(hidden.squeeze(0))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model = RNN(INPUT_DIM,EMBEDDING_DIM,HIDDEN_DIM,\
            OUTPUT_DIM,N_LAYERS,BIDIRECTIONAL,DROPOUT,PAD_IDX)
logger.info((f'The model has {count_parameters(model):,} trainable parameters'))

pretrained_embeddings = TEXT.vocab.vectors  #
model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
if USE_CUDA: model = model.to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.BCEWithLogitsLoss()
if USE_CUDA: criterion = criterion.to(device)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc  = 0
    model.train()

    for batch in iterator:
        predictions = model(batch.text).squezze(1)
        loss = criterion(predictions,batch.label)
        acc = binary_accuracy(predictions,batch.label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    model.train()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


N_EPOCHS = 5
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(model, train_iter, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), SAVE_LOG)

    logger.info(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    logger.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    logger.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')



model.load_state_dict(torch.load(SAVE_MODEL))
test_loss, test_acc = evaluate(model, test_iter, criterion)
logger.info(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')