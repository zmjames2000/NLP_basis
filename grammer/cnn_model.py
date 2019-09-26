# encoding=utf-8
import tensorflow as tf

class TCNNConfig(object):
    embedding_dim = 128  # 词向量维度
    seq_length    = 600  # 序列长度
    num_classes   = 10   # 类别数
    num_filters   = 256  # 卷积核数目
    kernel_size   = 5    # 卷积核尺寸
    vocab_size    = 5000 # 词汇表达小

    hidden_dim    = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate  = 1e-3  # 学习率

    batch_size     = 64  # 每批训练大小
    num_epochs     = 10  # 总迭代轮次
    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

class TextCNN(object):
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32,   [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        with tf.device('/gpu:1'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim]) #5000*128
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope('cnn'):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv') #
            # global max pooling layer
            gmp  = tf.reduce_mean(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope('score'):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='f1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1) # 预测类别

        with tf.name_scope('optimizer'):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y,1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
