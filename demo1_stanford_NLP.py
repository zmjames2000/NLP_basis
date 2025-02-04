# coding=utf-8

from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(r'D:\NLP_sourceCode\stanfordcorenlp')
# 如果所有设置都没有问题还是报错：
# 请注意：D:\Anaconda3\Lib\site-packages\stanfordcorenlp\corenlp.py
# memory 默认是4g，但是我只有8g，运行剩余不够4g，所以需要改小，但是速度会很慢，所以只有加内存条。

# step.1 启动 server
# Run a server using Chinese properties
# java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-chinese.properties -port 9000 -timeout 15000
# nlp = StanfordCoreNLP('http://localhost', port=9000)

sentence = 'Guangdong University of Foreign Studies is located in Guangzhou'

# print (nlp.word_tokenize(sentence))
# print (nlp.pos_tag(sentence))
# print (nlp.ner(sentence))
print (nlp.parse(sentence))  # 语法树
# print (nlp.dependency_parse(sentence)) #依存句法

nlp.close() #释放，否则后端服务器将消耗大量内存