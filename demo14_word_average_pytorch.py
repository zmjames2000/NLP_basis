#-*- coding:utf-8 -*-

import torch
import torch.nn.functional as F
import time,os,random

import torchtext
from torchtext import datasets

SEED = 1000

TEXT = torchtext.data.Field(tokenize='spacy') # pip install -U spacy  python -m spacy download en
LABEL= torchtext.data.LabelField(dtype=torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT,LABEL)
train_data, valid_data = train_data.split(random_state = random.seed(SEED))

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')
# print(vars(train_data.examples[0]))

TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)
# print(TEXT.vocab.freqs.most_common(20))
# print(TEXT.vocab.itos[:10])
# print(LABEL.vocab.stoi)

USE_CUDA = torch.cuda.is_available()
device = torch.device( 'cuda' if USE_CUDA  else 'cpu')

torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

BATCH_SIZE = 64
DATA_PATH  = r'./data/demo14'
SAVE_MODEL = DATA_PATH + os.sep + 'loss_model.pth'

train_iterator, valid_iterator, test_iterator = torchtext.data.BucketIterator.splits(dataset=(train_data, valid_data,test_data),
                                                                                     batch_size=BATCH_SIZE,
                                                                                     device=device)

class WordAVGModel(torch.nn.Module):
    def __init__(self, vocab_size,embedding_size,output_size,pad_idx):
        super(WordAVGModel,self).__init__()
        self.embed = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.linear = torch.nn.Linear(embedding_size, output_size)

    def forward(self,text):
        embedded = self.embed(text) # seg_len, batch_size, embedding_size
        embedded = embedded.permute(1,0,2) # 重新排序 比view好用多了  batch_size ,seg_len, embedding_size
        pooled   = F.avg_pool2d(embedded, kernel_size=(embedded.shape[1], 1)).squeeze() # batch_size ,1, embedding_size --》 squeeze()  [batch_size, embedding_size]
        #  kernel_size=(embedded.shape[1], 1)  窗口的大小.
        # seueeze() 表示把1维的都去除，squeeze(x,1) 去除指定维度， 只能去除1维的
        return  self.linear(pooled)

def count_parameters(model):
    return sum( p.numel() for p in model.parameters() if p.requires_grad) # numel 统计

VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_SIZE = 100
OUTPUT_SIZE = 1
PAD_IDX = TEXT.vocab.stoi(TEXT.pad_token)
UNK_IDX = TEXT.vocab.stoi(TEXT.unk_token)
learning_rate = 0.002

# glove 初始化模型
model = WordAVGModel(VOCAB_SIZE,EMBEDDING_SIZE,OUTPUT_SIZE,PAD_IDX)
print(count_parameters(model))

# 把模型初始化 glove的形状
pretained_embedding = TEXT.vocab.vectors
model.embed.weight.data.copy_(pretained_embedding)

model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.BCEWithLogitsLoss() # 不许要 sigmoid   而分类的 crossentryloss
if USE_CUDA:
    model = model.to(device)
    criterion = criterion.to(device)

def binary_accuarcy(preds, y):
    rouneded_preds = torch.round(torch.sigmoid(preds))
    correct = (rouneded_preds == y).float() # ture.float() false.float()
    acc = correct.sum()/len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc  = 0
    model.train()

    for batch in iterator:
        predicitions = model(batch.text).squezze(1) #
        loss = criterion(predicitions, batch.label)
        acc  = binary_accuarcy(predicitions, batch.label)
        #SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss/len(iterator), epoch_acc/len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc  = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc  = binary_accuarcy(predictions, batch.label)

            # no need SGD
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    model.train()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return  elapsed_mins, elapsed_secs

N_EPOCHS = 10
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), SAVE_MODEL)

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')