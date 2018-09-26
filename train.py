import tensorflow as tf
import cifar10
from datetime import datetime
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'cifar10_train',
                           '''Directory where to write event logs and checkpoint''')
tf.app.flags.DEFINE_integer('max_steps', 100000, '''Number of batches to run''')
tf.app.flags.DEFINE_boolean('log_device_placement', False, '''Whether to log device placement''')
tf.app.flags.DEFINE_integer('log_frequency', 10, '''How ofter to log results to the console''')


def train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()

        logits = cifar10.inference(images)
        loss = cifar10.loss(logits, labels)
        train_op = cifar10.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            '''
            该类用来打印训练信息
            '''
            def begin(self):
                '''
                在创建会话之前调用，调用begin()时，default graph
                会被创建，可在此处向default graph增加新op, begin()
                调用后，default graph不能再被掉用
                '''
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                '''
                调用在每个sess.run()执行之前，可以返回一个
                tf.train.SessRunArgs(op/tensor),在即将运行的会话中加入这些
                op/tensor; 加入的op/tensor会和sess.run()中已定义的op/tensor
                合并，然后一起执行。
                ＠param run_context: A 'SessionRunContext' object
                ＠return: None or a 'SessionRunArgs' object
                '''
                self._step += 1
                # 在这里返回你想在运行过程中产看的信息，以list的形式传递,如:[loss, accuracy]
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                '''
                调用在每个sess.run()之后，参数run_values是before_run()中要求的
                op/tensor的返回值;　
                可以调用run_contex.request_stop()用于停止迭代。　
                sess.run抛出任何异常after_run不会被调用
                ＠param run_context: A 'SessionRunContext' object
                ＠param run_values: A SessionRunValues object
                '''
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    # results返回的是上面before_run()的返回结果，上面是loss所以返回loss
                    # 如若上面返回的是个list,则返回的也是个list
                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    print('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                          % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))
        '''
        将计算图的各个节点/操作定义好，构成一个计算图。然后开启一个
        MonitoredTrainingSession来初始化/注册我们的图和其他信息
        在其参数hooks中，传递了三个hook:
        1. tf.train.StopAtStepHook(last_step):该hook是训练达到特定步数时请求
        停止。使用该hook必须要预先定义一个tf.train.get_or_create_global_step()
        2. tf.train.NanTensorHook(loss):该hook用来检测loss, 若loss的结果为NaN,则会
        抛出异常
        3. _LoggerHook():该hook是自定义的hook，用来检测训练过程中的一些数据，譬如loss, accuracy
        。首先会随着MonitoredTrainingSession的初始化来调用begin()函数，在这里初始化步数，before_run()
        函数会随着sess.run()函数的调用而调用。所以每训练一步调用一次，这里返回想要打印的信息，随后调用
        after_run()函数。
        '''
        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.train_dir,
                                               hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                                                      tf.train.NanTensorHook(loss),
                                                      _LoggerHook()],
                                               config=tf.ConfigProto(
                                                   log_device_placement=FLAGS.log_device_placement
                                               )) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
