# coding=utf-8

from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(r'D:\NLP_sourceCode\stanfordcorenlp',lang='zh')

# step.1 启动 server
# Run a server using Chinese properties
# java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-chinese.properties -port 9000 -timeout 15000
# nlp = StanfordCoreNLP('http://localhost', port=9000)

sentence = '清华大学位于北京。'

print (nlp.word_tokenize(sentence))
print (nlp.pos_tag(sentence))
print (nlp.ner(sentence))
print (nlp.parse(sentence))
print (nlp.dependency_parse(sentence))

nlp.close()