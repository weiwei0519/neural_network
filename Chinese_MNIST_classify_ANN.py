# coding=UTF-8
# 使用ANN对Chinese MNIST数据集进行图像分类的神经网络模型

'''
File Name: Chinese_MNIST_classify_ANN.py
Program IDE: PyCharm
Created Time: 2022/7/6 0006 22:13
Author: Wei Wei
'''

import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from util.pathutil import PathUtil
import numpy as np
import idx2numpy as idxnp

# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 默认设置，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)
project_path = PathUtil()
model_dir = project_path.rootPath + '/model/Chinese_MNIST'

# 加载Chinese MNIST数据集
dataset_dir = project_path.rootPath + '/datasets/Chinese_MNIST'
train_img_file = dataset_dir + '/train-images.idx3-ubyte'
train_label_file = dataset_dir + '/train-labels.idx1-ubyte'
val_img_file = dataset_dir + '/t10k-images.idx3-ubyte'
val_labels_file = dataset_dir + '/t10k-labels.idx1-ubyte'
pixel = (28, 28)
Chinese_MNIST_train_img = idxnp.convert_from_file(train_img_file)
Chinese_MNIST_train_label = idxnp.convert_from_file(train_label_file)
Chinese_MNIST_test_img = idxnp.convert_from_file(val_img_file)
Chinese_MNIST_test_label = idxnp.convert_from_file(val_labels_file)

epochs = 20  # 定义将数据输入模型的次数
batch_size = 64  # 定义每个批次输入数据量的大小
n_labels = Chinese_MNIST_train_label.shape[0]  # 分类集大小

# 将数据集归一化为[0，1]，并转换数据类型
Chinese_MNIST_train_img = tf.cast(Chinese_MNIST_train_img / 255.0, dtype=tf.float32)
Chinese_MNIST_test_img = tf.cast(Chinese_MNIST_test_img / 255.0, dtype=tf.float32)
Chinese_MNIST_train_label = tf.cast(Chinese_MNIST_train_label, dtype=tf.int64)
Chinese_MNIST_test_label = tf.cast(Chinese_MNIST_test_label, dtype=tf.int64)

# show example 数据
plt.figure(figsize=(32, 32))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(Chinese_MNIST_train_img[i], cmap=plt.cm.binary)
    plt.xlabel([Chinese_MNIST_train_label[i]])
plt.show()