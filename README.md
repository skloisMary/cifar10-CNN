# cifar10-CNN
卷积神经网络CNN在cifar10上的应用

本人进行了分别是在单个GPU和多个GPU下进行了测试，所以代码就被分成了两部分
单个GPU包含以下文件(在官方源码基础上进行更改)，见博客[卷积神经网络：CIFAR-10训练和测试（单块GPU）](https://blog.csdn.net/weixin_42111770/article/details/81940601):
|  文件  | 作用 |
|  cifar10_input.py  |读取本地CIFAR-10的二进制文件格式，定义函数distorted_inputs获得训练数据和inputs函数获取测试数据|
|  cifar10_model.py  |建立卷积神经模型，定义损失函数，训练器和正确率计算等函数|
|   cifar10_train_eval.py |训练CIFAR-10和评估CIFAR-10模型|

多个GPU包含以下文件(官方源码＋本人注释和理解)，见博客[官方卷积神经网络cifar10源码的学习笔记（多块GPU）](https://blog.csdn.net/weixin_42111770/article/details/82685668):
|  文件  | 作用|
|  input.py  | 读取本地CIFAR-10的二进制文件格式的内容。|
|   cifar10.py | 建立CIFAR-10的模型。|
| train.py | 在CPU或GPU上训练CIFAR-10的模型。|
|  eval.py | 评估CIFAR-10模型的预测性能。|
| multi_gpu_train.py | 在多GPU上训练CIFAR-10的模型。|
