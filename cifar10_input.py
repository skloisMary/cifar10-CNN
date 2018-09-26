# coding:utf-8
"""
CIFAR-10 该数据集共有60000张32*32*3大小的图像，分为10类，每类6000张图
其中50000张用于训练，构成5个训练批，每一批次10000张图，10000张用于测试，单独构成一批
"""
import tensorflow as tf
import os

IMAGE_SIZE = 24

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
    """从CIFAR10数据集种读取数据
    @param filename_queue: 要读取的文件名队列
    @return: 某个对象，具有以下字段
             height：图片高度
             width：图片宽度
             depth：图片深度
             key： 一个描述当前抽样数据的文件名和记录数地标量字符串
             label： 一个int32类型的标签， 取值0...9
             uint8image: 一个[height, width, depth]维度的图像数据
    """
    # 建立一个空类， 方便数据的结构化储存
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_byte = result.height * result.width * result.depth
    record_bytes = label_bytes + image_byte

    # tf.FixedLengthRecordReader读取固定长度字节数信息，下次调用时会接着上次读取的位置继续读取文件
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    # decode_raw操作将一个字符串转换成一个uint8的张量
    record_bytes = tf.decode_raw(value, tf.uint8)
    # tf.strides_slice(input, begin, end, strides=None)截取[begin, end)之间的数据
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes+image_byte]),
                             [result.depth, result.height, result.width])
    # convert from [depth, height, width] to [height, width, depth]
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result


# 获取训练数据
def distorted_inputs(data_dir, batch_size):
    """对cifar训练集中的image数据进行变换，图像预处理
    param data_dir: 数据所处文件夹名称
    param batch_size: 批次大小
    return:
           images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
           labels: 1D tensor of [batch_size] size
    """
    filename = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    for f in filename:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filename)

    # 数据扩增
    with tf.name_scope('data_augmentation'):
        read_input = read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # tf.random_crop 对输入图像进行随意裁剪
        distored_image = tf.random_crop(reshaped_image, [height, width, 3])
        # tf.image.random_flip_left_right 随机左右翻转图片
        distored_image = tf.image.random_flip_left_right(distored_image)
        # tf.image.random_brightness在某范围随机调整图片亮度
        distored_image = tf.image.random_brightness(distored_image, max_delta=63)
        # tf.image.random_contrast 在某范围随机调整图片对比度
        distored_image = tf.image.random_contrast(distored_image, lower=0.2, upper=1.8)
        # 归一化， 三维矩阵中的数字均值为0，方差为1， 白话操作
        float_image = tf.image.per_image_standardization(distored_image)

        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    image_batch, label_batch = tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size,
                                                      capacity= min_queue_examples + 3 * batch_size,
                                                      min_after_dequeue=min_queue_examples)
    tf.summary.image('image_batch_train', image_batch)
    return image_batch, tf.reshape(label_batch, [batch_size])


# 获取测试数据
def inputs(data_dir, batch_size):
    """
    输入
    param data_dir: 数据所处文件夹名称
    param batch_size: 批次大小
    return:
     images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
     labels: 1D tensor of [batch_size] size
    """
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames)
        read_input = read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE
        # 剪裁或填充，会根据原图像的尺寸和指定目标图像的尺寸选择剪裁还是填充，如果原图像尺寸大于目标图像尺寸
        # 则在中心位置剪裁，反之则用黑色像素填充
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)

        # 归一化， 三维矩阵中的数字均值为0，方差为1， 白话操作
        float_image = tf.image.per_image_standardization(resized_image)
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
    image_batch, label_batch = tf.train.batch([float_image, read_input.label], batch_size=batch_size,
                                              capacity=min_queue_examples + 3 * batch_size)
    tf.summary.image('image_batch_evaluation', image_batch)
    return image_batch, tf.reshape(label_batch, [batch_size])