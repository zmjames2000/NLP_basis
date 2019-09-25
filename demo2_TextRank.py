# econding: utf-8

from jieba import analyse
# 引入TextRank关键词抽取接口
textrank = analyse.textrank

raw_text =  "非常线程是程序执行时的最小单位，它是进程的一个执行流，\ 是CPU调度和分派的基本单位，一个进程可以由很多个线程组成，\ 线程间共享进程的所有资源，每个线程有自己的堆栈和局部变量。\ 线程由CPU独立调度执行，在多CPU环境下就允许多个线程同时运行。\ 同样多线程也可以实现并发操作，每个请求分配一个线程来处理。"
# 基于TextRank算法进行关键词抽取
keywords = textrank(raw_text,topK=len(raw_text), withWeight=True, allowPOS=('ns','n','a'))
#topK=10 表示前面10个词  allowPOS 设置词性
"""
Parameter:
    - topK: return how many top keywords. `None` for all possible words.
    - withWeight: if True, return a list of (word, weight);
                  if False, return a list of words.
    - allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v'].
                if the POS of w is not in this list, it will be filtered.
    - withFlag: if True, return a list of pair(word, weight) like posseg.cut
                if False, return a list of words
"""
print(keywords)
# [('线程', 1.0), ('单位', 0.918659371013185), ('进程', 0.7741356100393552), ('基本', 0.6195197989254972), ('程序执行', 0.4975135665898287), ('调度', 0.4877874375629275), ('分派', 0.4735514079856176), ('局部变量', 0.37617102712465794), ('堆栈', 0.3744186029702699), ('最小', 0.341143278070699), ('资源', 0.33931958742765966)]

# 输出抽取出的关键词
words=[keyword+'/' for keyword,w in keywords if w>0.2]
# 线程/ 单位/ 进程/ 基本/ 程序执行/ 调度/ 分派/ 局部变量/ 堆栈/ 最小/ 资源/
print (' '.join(words) + "\n")



