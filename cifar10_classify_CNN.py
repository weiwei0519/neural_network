# coding=UTF-8
# 基于tensorflow的CNN实现
# 用于cifar10数据集训练

'''
File Name: cifar10_classify_CNN.py
Program IDE: PyCharm
Created Time: 2022/6/19 0019 21:23
Author: Wei Wei
'''

import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from util.pathutil import PathUtil

# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 默认设置，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

tf.random.set_seed(2345)
project_path = PathUtil()
model_dir = project_path.rootPath + '/model/cifar10_CNN'

# 加载CIFAR10数据集
cifar10 = tf.keras.datasets.cifar10
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
# plt.figure(figsize=(32, 32))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(cifar10_train_x[i], cmap=plt.cm.binary)
#     plt.xlabel([cifar10_train_y[i]])
# plt.show()

# 模型加载
try:
    model = load_model(model_dir)
except IOError:
    print("model file doesn't exist! Need re-train!")
    # 构建Sequential模型层
    # 如下为基于FNN
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    #     tf.keras.layers.Dense(3072, activation=tf.nn.relu),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    # ])
    # 如下为基于CNN
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=(32, 32, 3), kernel_size=(3, 3), filters=64, strides=1, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(3072, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(units=n_labels, activation='softmax')
    ])

    # 模型可视化
    model.summary()

    # 定义模型优化器
    optimizer = tf.keras.optimizers.Adam()
    # 定义损失函数
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    # 定义模型评估方法: 混淆矩阵
    accuracy = 'sparse_categorical_accuracy'

    # 模型编译
    model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])

    # 输入数据集，进行模型训练
    history = model.fit(cifar10_train_x, cifar10_train_y, batch_size=batch_size, epochs=epochs,
                        validation_data=(cifar10_test_x, cifar10_test_y), validation_freq=1, verbose=2)
    '''
    verbose：日志显示
    verbose = 0 不在标准输出流输出日志信息
    verbose = 1 输出进度条记录
    verbose = 2 每个epoch输出一行记录
    '''

    # 显示训练集和验证集的acc和loss曲线
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

# 模型评估
model.evaluate(cifar10_test_x, cifar10_test_y, verbose=2)

# 模型持久化
model.save(model_dir)
