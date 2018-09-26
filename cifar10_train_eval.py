# coding:utf-8
import tensorflow as tf
import numpy as np
import cifar10_input
import cifar10_model
import matplotlib.pyplot as plt
import os

# 使用GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

EPOCHES = 50000
DATA_DIR = cifar10_model.DATA_DIR
BATCH_SIZE = cifar10_model.BATCH_SIZE


# 训练
def train():
    # 读取图片并带入网络计算
    images, labels = cifar10_input.distorted_inputs(DATA_DIR, BATCH_SIZE)
    t_logits = cifar10_model.inference(images)
    # 损失值
    t_loss = cifar10_model.loss(t_logits, labels)
    tf.summary.scalar('loss_value', t_loss)
    # 优化器
    global_step = tf.Variable(0, trainable=False)
    t_optimizer = cifar10_model.train_step(t_loss, global_step)
    # 准确值
    t_accuracy = cifar10_model.accuracy(t_logits, labels) # 训练集正确率计算
    tf.summary.scalar('accuracy_value', t_accuracy)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    Accuracy_value = []
    Loss_value = []
    # 设定定量的GPU显存使用量
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        train_writer = tf.summary.FileWriter('signal_GPU/logs', session.graph)
        for index in range(EPOCHES):
            _, loss_value, accuracy_value, summary = session.run([t_optimizer, t_loss, t_accuracy, merged])
            Accuracy_value.append(accuracy_value)
            Loss_value.append(loss_value)
            if index % 1000 == 0:
                print('index:', index, ' loss_value:', loss_value, ' accuracy_value:', accuracy_value)
            train_writer.add_summary(summary, index)
        saver.save(session, os.path.join('signal_GPU/saver', 'model.ckpt'))
        # accuracy value
        plt.figure(figsize=(20, 10))
        plt.plot(range(EPOCHES), Accuracy_value)
        plt.xlabel('training step')
        plt.ylabel('accuracy value')
        plt.title('the accuracy value of training data')
        plt.savefig('signal_GPU/results/accuracy.png')
        # loss value
        plt.figure()
        plt.plot(range(EPOCHES), Loss_value)
        plt.xlabel('training value')
        plt.ylabel('loss value')
        plt.title('the value of the loss function of the training data')
        plt.savefig('signal_GPU/results/loss.png')
        #
        train_writer.close()
        coord.request_stop()
        coord.join(threads)


# 验证
def evaluation():
    with tf.Graph().as_default():
        n_test = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        eval_images, eval_lables = cifar10_input.inputs(DATA_DIR, BATCH_SIZE)
        eval_logits = cifar10_model.inference(eval_images)
        # tf.nn.in_top_k(predictions, targets, k, name=None)
        # 每个样本的预测结果的前k个最大的数里面是否包括包含targets预测中的标签，一般取1，
        # 即取预测最大概率的索引与标签的对比
        top_k_op = tf.nn.in_top_k(eval_logits, eval_lables, 1)
        saver = tf.train.Saver()
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state('signal_GPU/saver')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            num_iter = int(n_test / BATCH_SIZE)
            true_count = 0
            for step in range(num_iter):
                predictions = session.run(top_k_op)
                true_count = true_count + np.sum(predictions)
            precision = true_count / (num_iter * BATCH_SIZE)
            print('precision=', precision)
            coord.request_stop()
            coord.join(threads)


def main(argv=None):
    train()
    evaluation()


if __name__ == '__main__':
    tf.app.run()
