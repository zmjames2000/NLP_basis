#-*- coding:utf-8 -*-

import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr

def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open(filename,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open(vocab_dir, 'w', encoding='utf-8',errors='ignore').write('\n'.join(words)+'\n')

def read_vocab(vocab_dir):
    """读取词汇表"""
    with open(vocab_dir,'r',encoding='utf-8',errors='ignore') as fp:
        words = [ _.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(0, len(words))))
    return words, word_to_id

def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    # categories = [ x for x in categories]
    cat_to_id = dict(zip(categories, range(0, len(categories)))) #{'体育':0, '财经':1 ...}
    return categories, cat_to_id

def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度  ( 将每句话都补齐至600）
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)  # 将句子都变成600大小的句子，超过600的从后边开始数，去除前边的
    # x_pad  50000行 600 列
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]