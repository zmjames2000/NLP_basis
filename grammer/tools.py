# encoding=utf-8

import os,gc,re,sys
from itertools import chain
from stanfordcorenlp import StanfordCoreNLP
import jpype

def getJVMClass(packages):#JVM只需要调用一次就可以了
    # 启动JVM
    # print ('jvmPath:{}'.format(jpype.getDefaultJVMPath())) #jvmPath:D:\jdk-12.0.2\bin\server\jvm.dll
    jvmPath = jpype.getDefaultJVMPath()#如果获取不到，可以用绝对路径，本地找一下
    # 加载jar包，获取jar所在的路径
    # print(jvmPath)
    root_path = r'D:\Anaconda3\Lib\site-packages\pyhanlp\static'
    djclass_path = "-Djava.class.path="+root_path+os.sep+'hanlp-1.7.4.jar;'+root_path
    jpype.startJVM(jvmPath, djclass_path) #D:\PyCharm\PyCharm 2019.2\bin\pycharm64.exe.vmoptions
    JDClass = jpype.JClass(packages)#包名.类名
    # 创建类实例对象
    jdc = JDClass()
    return  jdc

Tokenizer = getJVMClass('com.hankcs.hanlp.tokenizer.StandardTokenizer')
# NLPTokenizer = getJVMClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')
stanford_nlp = StanfordCoreNLP(r'D:/NLP_sourceCode/stanfordcorenlp/', lang='zh')
drop_pos_set = set(['xu','xx','y','yg','wh','wky','wkz','wp','ws','wyy','wyz','wb','u','ud','ude1','ude2','ude3','udeng','udh'])
han_pattern=re.compile(r'[^\dA-Za-z\u3007\u4E00-\u9FCB\uE815-\uE864]+')

def to_string(sentence, return_generator=False):
    if return_generator:
        return (word_pos_item.toString().split('/') for word_pos_item in Tokenizer.segment(sentence))
    else:
        return [(word_pos_item.toString().split('/')[0], word_pos_item.toString().split('/')[1]) for word_pos_item in Tokenizer.segment(sentence)]
        # return ' '.join([ word_pos_item.toString().split('/')[0] for word_pos_item in Tokenizer.segment(sentence)])
    # 这里的“”.split('/')可以将string拆分成list 如：'ssfa/fsss'.split('/') => ['ssfa', 'fsss']

def to_string_hanlp(sentence, return_generator=False):
    if return_generator:
        return ( word_pos_item.toString().split('/') for word_pos_item in Tokenizer.segment(sentence))
    else:
        return [(word_pos_item.toString().split('/')[0], word_pos_item.toString().split('/')[1]) for word_pos_item in Tokenizer.segment(sentence)]

def seg_sentences(sentence, with_filter=True, return_generator=False):
    segs = to_string(sentence, return_generator=return_generator)
    if with_filter:
        g = [word_pos_pair[0] for word_pos_pair in segs if len(word_pos_pair)==2 and word_pos_pair[0]!=' ' and word_pos_pair[1] not in drop_pos_set ]
    else:
        g = [word_pos_pair[0] for word_pos_pair in segs if len(word_pos_pair) == 2 and word_pos_pair[0] != ' ']
    return iter(g) if return_generator else g

def ner_stanford(raw_sentence, return_list=True):
    if len(raw_sentence.strip())>0:
        return stanford_nlp.ner(raw_sentence) if return_list else iter(stanford_nlp.ner(raw_sentence))

def ner_handlp(raw_sentence, return_list=True):
    if len(raw_sentence.strip())>0:
        return NLPTokenizer.segment(raw_sentence) if return_list else iter(NLPTokenizer.segment(raw_sentence))

def cut_stanford(raw_sentence, return_list=True):
    if len(raw_sentence.strip())>0:
        return stanford_nlp.pos_tag(raw_sentence) if return_list else iter(stanford_nlp.pos_tag(raw_sentence))

def cut_hanlp(raw_sentence,return_list=True):
    if len(raw_sentence.strip())>0:
        return to_string(raw_sentence) if return_list else iter(to_string(raw_sentence))