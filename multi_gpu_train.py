import tensorflow as tf
import numpy as np
import cifar10
import os
import re
import time
from datetime import datetime

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'cifar10_multi_gpu_train_logs',
                           '''Directory where to write event logs and checkpoint''')
tf.app.flags.DEFINE_integer('max_steps', 100000, '''Number of batches to run''')
tf.app.flags.DEFINE_integer('num_gpus', 1, '''How many GPUs to use''')
tf.app.flags.DEFINE_boolean('log_device_placement', False, '''Whether to log device placement''')


def tower_loss(scope, images, labels):
    '''
    当一个tower运行CIFAR模型时，计算total loss
    @param scope: 独特的前缀字符串表明CIFAR tower, 例如'tower_0'
    @param images:
    @param labels:
    @return: 一批次数据的total loss
    '''
    logits = cifar10.inference(images)
    _ = cifar10.loss(logits, labels)
    # 从当前tower中取出‘losses’的全部元素，构成一个列表
    losses = tf.get_collection('losses', scope)
    # tf.add_n([p1, p2, p3, ...])函数是实现一个列表元素的相加
    total_loss = tf.add_n(losses, name='total_loss')
    for l in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss


def average_gradients(tower_grads):
    '''
    计算平均梯度对在所有towers上的每一个共享变量
    ＠param tower_grads:　
    List of lists of (gradient, variable) tuples
    外一层的list　is over individual gradients
    内一层的list is over individual gradients
    ＠return:
    List of pairs of (gradient, variable) where the gradient has been
    averaged across all towers
    '''
    average_grads = []
    # zip()是Python的一个内建函数，它接受一系列可迭代的对象作为参数，将对象中
    # 对应的元素打包成一个个tuple（元祖），然后返回由这些tuple组成的list（列表）
    # 如果各个迭代器的元素不一致，则返回列表长度与最短的对象相同，利用*号操作符，可以将
    # 元祖解压为列表
    for grad_and_vars in zip(*tower_grads):
        # 每一个grad_and_vars的格式为:
        # ((grad0_gpu0, var0_gpu0), ... ,(grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # tf.expand_dims()为张量+1维
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        # 因为这些变量是要被共享在towers上的，所以这些变量是重复的
        # 因此返回第一个tower上的变量即可
        v = grad_and_vars[0][1]
        grad_and__var = (grad, v)
        average_grads.append(grad_and__var)

    return average_grads


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # 创建global_step，训练步数，在训练时，自动增加，　名称是global_step，　
        # shape是[],表示常数，初始值为0，非训练参数
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0), trainable=False)
        #
        num_batches_per_epoch = (cifar10.NUM_EXAMPLE_PER_EPOCH_FOP_TRAIN / FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

        lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        cifar10.LEARNING_RATE_DECAY_FACTOR, staircase=True)
        opt = tf.train.GradientDescentOptimizer(lr)

        images, labels = cifar10.distorted_inputs()
        # 使用预加载队列，获取batch_queue
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images, labels],
                                                                    capacity=2 * FLAGS.num_gpus)
        tower_grads = []
        # 变量的名称
        with tf.variable_scope(tf.get_variable_scope()):
            # 创建GPU的循环
            for i in range(FLAGS.num_gpus):
                # 指定GPU
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
                        image_batch, label_batch = batch_queue.dequeue()
                        loss = tower_loss(scope, image_batch, label_batch)
                        # 重用变量
                        tf.get_variable_scope().reuse_variables()
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        summaries.append(tf.summary.scalar('learning_rate', lr))

        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        variable_average = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY, global_step)
        variable_average_op = variable_average.apply(tf.trainable_variables())

        # tf.group()将参数中的一个operation作为一个组，把这些操作合成一个操作
        train_op = tf.group(apply_gradient_op, variable_average_op)
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge(summaries)
        init = tf.global_variables_initializer()

        # tf.ConfigProto()函数在创建session的时候，用来对session进行参数配置
        # allow_soft_placement=True自行选择运行设备
        # log_device_placement设备指派情况，设置为True, 可以获取operations 和
        # Tensor被指派到哪个设备上运行，会在终端打印出各项操作是在哪个设备上运行的
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                print('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                      % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % 1000 == 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()

