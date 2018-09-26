import tensorflow as tf
from datetime import datetime
import numpy as np
import cifar10
import time
import math

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', 'cifar10_eval_logs',
                           '''Directory where to write event logs''')
tf.app.flags.DEFINE_string('eval_data', 'test',
                           '''Either 'test' or 'train_eval'.''')

tf.app.flags.DEFINE_string('checkpoint_dir', 'cifar10_train',
                           '''Directory where to read model checkpoint''')

tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            '''How often to run the eval.''')

tf.app.flags.DEFINE_integer('num_examples', 10000, '''NUmber of examples to run''')
tf.app.flags.DEFINE_boolean('run_once', False,
                            '''Whether to run eval only once''')


def eval_once(saver, summary_writer, top_k_op, summary_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord,
                                                 daemon=True, start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    with tf.Graph().as_default() as g:
        eval_data = FLAGS.eval_data == 'test'
        images, labels = cifar10.inputs(eval_data=eval_data)

        logits = cifar10.inference(images)

        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        variable_average = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY
        )
        variable_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
