# NLP_basis
【2019最新AI 自然语言处理之深度机器学习顶级项目实战课程】做的笔记与代码

---

###  demo1_jieba_load_userdict.py
######  测试 及 演示jieba的分词
###  demo1_hanlp.py
######  测试 及 演示hanlp的分词
###  demo1_stanford_NLP.py
######  测试 及 演示stanfordcoreNLP的分词
###  demo2_stanfordnlp_ner_test.py
######  太耗内存
###  demo2_text_rank_demo
节点的权重不仅依赖于它的入度结点，还依赖于这些入度结点的权重，入度结点越多，入度结点的权重越大，说明这个结点的权重越高；图中任两点 Vi , Vj 之间边的权重为 wji , 
对于一个给定的点 Vi, In(Vi) 为 指 向 该 点 的 点 集 合 , Out(Vi) 为点 Vi 指向的点集合。

####  **应用到关键短语抽取**：
1. 预处理，首先进行分词和词性标注，将单个word作为结点添加到图中；
2. 设置语法过滤器，将通过语法过滤器的词汇添加到图中；出现在一个窗口中的词汇之间相互形成一条边；
3. 基于上述公式，迭代直至收敛；一般迭代20-30次，迭代阈值设置为0.0001；
4. 根据顶点的分数降序排列，并输出指定个数的词汇作为可能的关键词；
5. 后处理，如果两个词汇在文本中前后连接，那么就将这两个词汇连接在一起，作为关键短语；

---
-  Word2vec 的Skip-gram 和 CBOW 模型
-  如果是用一个词语作为输入,来预测它周围的上下文,那这个模型叫做 『Skip-gram 模型』
-  而如果是拿一个词语的上下文输入,来预测这个词语本身,则是 『CBOW 模型』

###  demo6 word2vec skip_garm
-  [理解 Word2Vec 之 Skip-Gram 模型](https://zhuanlan.zhihu.com/p/27234078)
-  [基于TensorFlow实现Skip-Gram模型](https://zhuanlan.zhihu.com/p/27296712)
###  demo6 word2vec cbow
A、Continuous Bag of Words Model(CBOW)
-  给定上下文预测目标词的概率分布,例如,给定{The,cat,(),over,the,puddle}预测中心词是jumped的概率,模型的结构
###  demo7 CNN 
[https://zhuanlan.zhihu.com/p/28087321](https://zhuanlan.zhihu.com/p/28087321)









