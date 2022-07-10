# coding=UTF-8
# 基于tensorflow的深度前馈神经网络的实现
# 用于cifar100数据集训练

'''
File Name: cifar100_classify.py
Program IDE: PyCharm
Created Time: 2022/6/27 0027 21:32
Author: Wei Wei
'''

import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from util.pathutil import PathUtil


# 加载CIFAR10数据集
cifar100 = tf.keras.datasets.cifar100
(cifar10_train_x, cifar10_train_y), (cifar10_test_x, cifar10_test_y) = cifar100.load_data()
