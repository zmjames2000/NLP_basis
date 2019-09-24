# -*- encoding: utf-8 -*-
# encoding: unicode_escape

import jieba
import re
from grammer.rules import grammer_parse

with open('./data/test.txt','r',encoding='gbk') as fp,\
    open('./data/out_test.txt','w',encoding='gbk') as fout:
    [grammer_parse(line.strip(), fout) for line in fp if len(line.strip())>0]

if __name__ == '__main__':
    pass
