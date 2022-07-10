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

# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 默认设置，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)
project_path = PathUtil()
model_dir = project_path.rootPath + '/model/Chinese_MNIST'
dataset_dir = project_path.rootPath + '/datasets/Chinese_MNIST'
pixel = (28, 28)

def load_data(dataset_dir, pixel):


# 加载CIFAR10数据集
Chinese_MNIST = tf.keras.datasets.cifar10
(cifar10_train_x, cifar10_train_y), (cifar10_test_x, cifar10_test_y) = cifar10.load_data()

epochs = 20  # 定义将数据输入模型的次数
batch_size = 64  # 定义每个批次输入数据量的大小
n_labels = 10  # 分类集大小

# 将数据集归一化为[0，1]，并转换数据类型
cifar10_train_x = tf.cast(cifar10_train_x / 255.0, dtype=tf.float32)
cifar10_test_x = tf.cast(cifar10_test_x / 255.0, dtype=tf.float32)
cifar10_train_y = tf.cast(cifar10_train_y, dtype=tf.int64)
cifar10_test_y = tf.cast(cifar10_test_y, dtype=tf.int64)

# show example 数据
plt.figure(figsize=(32, 32))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(cifar10_train_x[i], cmap=plt.cm.binary)
    plt.xlabel([cifar10_train_y[i]])
plt.show()