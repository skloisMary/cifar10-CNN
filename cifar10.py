import tensorflow as tf
import os
import re
import input
import sys
import tarfile
from six.moves import urllib

# tf.app.flags主要处理命令行参数的解析工作
FLAGS = tf.app.flags.FLAGS

# tf.app.flag.DEFINE_xxx()就是添加命令行的可选参数
tf.app.flags.DEFINE_integer('batch_size', 128, '''Number of images to process in  a batch''')
tf.app.flags.DEFINE_string('data_dir', 'cifar10_data',
                           '''Path to the CIFAR-10 data directory''')
tf.app.flags.DEFINE_boolean('use_fp16', False, '''Train the model using fp16''')

# 参数设置
IMAGE_SIZE = input.IMAGE_SIZE
NUM_CLASS = input.NUM_CLASSES
NUM_EXAMPLE_PER_EPOCH_FOP_TRAIN = input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLE_PER_EPOCH_FOR_TEST = input.NUM_EXAMPLES_PER_EPOCH_FOR_TEST


MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
# 衰减系数decay_rate
LEARNING_RATE_DECAY_FACTOR = 0.1
# 初始学习率
INITIAL_LEARNING_RATE = 0.1

TOWER_NAME = 'tower'
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


# 为了在多个GPU上共享变量，所有的变量都绑定在CPU上，并通过tf.get_variable()访问
def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


# weight decay 是放在正则项前面的一个系数，控制正则化在损失函数中所占的权重，正则项一般表示模型的复杂度，
# 所以weight decay的作用
# 是调节模型复杂度对损失函数的影响
def _variable_with_weight_decay(name, shape, stddev, wd):
    '''用weight decay 建立一个初始的变量
    @param name:
    @param shape:
    @param stddev: 截断高斯分布的标准偏差
    @param wd: 如果wd不为None, 为变量添加L2_loss并与权重衰减系数相乘
    @return: 张量
    '''
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float16
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        # 添加L2Loss, 并将其添加到‘losses’集合
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


# 训练时输入函数
def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    print(FLAGS.batch_size)
    images, labels = input.distorted_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


# 验证时输入函数
def inputs(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dia')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = input.inputs(eval_data, data_dir, FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


# 卷积网络架构
def inference(images):
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weight', shape=[5, 5, 3, 64], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_1')

    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(norm1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_2')

    # local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    # linear layer
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASS], stddev=1 / 192.0, wd=None)
        biases = _variable_on_cpu('biases', [NUM_CLASS], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


# 损失函数
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                   logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # tf.add_to_collection 把变量放入一个集合，把很多变量变成一个列表
    tf.add_to_collection('losses', cross_entropy_mean)
    # tf.get_collection: 从一个集合中取出全部变量，是一个列表
    # 'losses'集合中包括‘cross_entropy_mean’和'weight_decay'
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# 对total_loss生成滑动均值和汇总，通过使用指数衰减，来维护变量的滑动均值(Moving Average)
def _add_loss_summaries(total_loss):
    loss_average = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_average_op = loss_average.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + '(raw)', l)
        tf.summary.scalar(l.op.name, loss_average.average(l))
    return loss_average_op


def train(total_loss, global_step):
    '''
    @param total_loss: loss() 函数的返回变量
    @param global_step: 一个记录训练步数的整数变量
    @return:
    '''
    num_batches_per_epoch = NUM_EXAMPLE_PER_EPOCH_FOP_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY) # 衰减速度
    # 指数衰减
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR, staircase=True)
    tf.summary.scalar('learning_rate', lr)

    loss_average_op = _add_loss_summaries(total_loss)
    # tf.control_dependencies()设计是用来控制计算流图的，给图中的某些计算指定顺序
    # tf.control_dependencies()是一个context manager，　控制节点执行顺序
    # 先执行[]中的操作，在执行context中的操作
    with tf.control_dependencies([loss_average_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        # 利用computer_gradients()函数计算梯度
        grads = opt.compute_gradients(total_loss)

    # 调用apply_gradients()函数来更新该梯度所对应的参数的状态
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    # tf.train.ExponentialMovingAverage(decay, steps就是采用滑动平均的方法更新参数。
    # 这个函数初始化需要提供一个衰减速率decay，用于控制模型的更新速度，decay越大越趋于稳定
    # ExponentialMovingAverage还提供num_updates参数来设置decay的大小，使得模型在训练的
    # 初始阶段更新得更快
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, num_updates=global_step)
    # apply()方法添加了训练变量的影子副本，并保持其影子副本中训练变量的移动平均值操作
    # 在每次训练之后调用此操作，更新移动平均值
    with tf.control_dependencies([apply_gradient_op]):
        variable_average_op = variable_average.apply(tf.trainable_variables())

    return variable_average_op


def maybe_download_and_extract():
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>>Downloading %s %.1f%%'
                             % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory,
                                      'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)