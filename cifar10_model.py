# coding:utf-8
"""
构建CIFAR-10网络
"""
import tensorflow as tf
import cifar10_input

# 参数
INITIAL_LEARNING_RATE = 0.1
DECAY_RATE = 0.96
DECAY_STEP = 300

IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


BATCH_SIZE = 128
DATA_DIR = '/home/mary/PycharmProjects/cifar10/cifar10_data/cifar-10-batches-bin'


def inference(images):
    """Build the CIFAR-10 model
    conv1-->pooling1-->norm1-->conv2-->pooling_2-->norm_2-->local3-->local4-->softmax_linear
    @param images: 喂入的数据[batch_size, height, width, 3]
    @return: logits: 预测向量
    """
    # conv1
    with tf.name_scope('conv1'):
        conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], mean=0, stddev=5e-2), name='weights')
        conv1_b = tf.Variable(tf.zeros(64), name='biases')
        conv1 = tf.nn.relu(tf.nn.conv2d(images, conv1_w, strides=[1, 1, 1, 1], padding='SAME') + conv1_b)
    tf.summary.histogram('conv1_w', conv1_w)
    tf.summary.histogram('conv1_b', conv1_b)

    # pooling_1
    pooling_1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling_1')
    # norm1
    norm1 = tf.nn.lrn(pooling_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm_1')

    # conv2
    with tf.name_scope('conv2'):
        conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], mean=0, stddev=5e-2), name='weights')
        conv2_b = tf.Variable(tf.zeros(64), name='biases')
        conv2 = tf.nn.relu(tf.nn.conv2d(norm1, conv2_w, strides=[1, 1, 1, 1], padding='SAME') + conv2_b)
    tf.summary.histogram('conv2_w', conv2_w)
    tf.summary.histogram('conv2_b', conv2_b)

    # pooling_2
    pooling_2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling_2')
    # norm 2
    norm2 = tf.nn.lrn(pooling_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm_2')

    #
    reshaped_output = tf.reshape(norm2, shape=[BATCH_SIZE, -1])
    dim = reshaped_output.shape[1].value

    # local 3
    with tf.name_scope('local3'):
        local3_w = tf.Variable(tf.truncated_normal(shape=[dim, 384], mean=0, stddev=0.04), name='weights')
        local3_b = tf.Variable(tf.zeros(384), name='biases')
        local3 = tf.nn.relu(tf.matmul(reshaped_output, local3_w) + local3_b)
    tf.summary.histogram('local3_w', local3_w)
    tf.summary.histogram('local3_b', local3_b)

    # local 4
    with tf.name_scope('local4'):
        local4_w = tf.Variable(tf.truncated_normal(shape=[384, 192], mean=0, stddev=0.04), name='weights')
        local4_b = tf.Variable(tf.zeros(192), name='biases')
        local4 = tf.nn.relu(tf.matmul(local3, local4_w) + local4_b)
    tf.summary.histogram('local4_w', local4_w)
    tf.summary.histogram('local4_b', local4_b)

    # softmax_linear
    with tf.name_scope('softmax_linear'):
        output_w = tf.Variable(tf.truncated_normal(shape=[192, 10], mean=0, stddev=1 / 192.0))
        output_b = tf.Variable(tf.zeros(10))
        logits = tf.nn.relu(tf.matmul(local4, output_w) + output_b)
    return logits


def loss(logits, labels):
    '''
    损失函数
    param logits: 预测向量
    param labels:  真实值
    return: 损失函数值
    '''
    with tf.name_scope('loss'):
        labels = tf.cast(labels, tf.int64)
        # logits通常是神经网络最后连接层的输出结果，labels是具体哪一类的标签
        # 这个函数是直接使用标签数据的，而不是采用one-hot编码形式
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        return cross_entropy_mean


# 训练优化器
def train_step(loss_value, global_step):
    # 指数衰减学习率
    '''
    tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=True/False)
    learning_rate: 初始的学习率
    global_step: 当前全局的迭代步数
    decay_steps: 每次迭代时需要经过多少步数
    decay_rate: 衰减比例
    staircase: 是否呈现阶梯状衰减
    '''
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step=global_step, decay_steps=DECAY_STEP,
                                               decay_rate=DECAY_RATE, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_value)
    return optimizer


# 计算正确率
def accuracy(logits, lables):
    reshape_logits = tf.cast(tf.argmax(logits, 1), tf.int32)
    accuracy_value = tf.reduce_mean(tf.cast(tf.equal(reshape_logits, lables), dtype=tf.float32))
    return accuracy_value