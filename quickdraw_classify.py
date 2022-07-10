# coding=UTF-8
# 基于tensorflow的VGG19深度卷积网络
# 用于quickdraw图像数据集分类训练

'''
File Name: quickdraw_classify.py
Program IDE: PyCharm
Created Time: 2022/7/2 0002 15:55
Author: Wei Wei
'''

import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from util.pathutil import PathUtil
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 默认设置，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)
project_path = PathUtil()
model_dir = project_path.rootPath + '/model/quickdraw'
datasets_dir = 'E:\\Development\\tensorflow_datasets\\quickdraw'

epochs = 25  # 定义将数据输入模型的次数
batch_size = 128  # 定义每个批次输入数据量的大小
pixels = (28, 28)
n_labels = 10  # 分类集大小

# 加载quickdraw数据集
# 使用 as_supervised=True，输出结果集为(features, label) 元组格式。
quickdraw_dataset, metadata = tfds.load(name="quickdraw_bitmap",
                                        download=True,
                                        data_dir=datasets_dir,
                                        with_info=True,
                                        as_supervised=True,
                                        split='train[:1%]')    # 只加载部分数据
(quickdraw_x, quickdraw_y) = quickdraw_dataset.take(1)

# 将数据集归一化为[0，1]，并转换数据类型
quickdraw_x = tf.cast(quickdraw_x / 255.0, dtype=tf.float32)
quickdraw_y = tf.cast(quickdraw_y, dtype=tf.int64)

# 训练集和测试集拆分
quickdraw_train_x, quickdraw_train_y, quickdraw_test_x, quickdraw_test_y = train_test_split(quickdraw_x, quickdraw_y,
                                                                                            test_size=0.2,
                                                                                            random_state=42)

# show example 数据
plt.figure(figsize=(32, 32))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(quickdraw_train_x[i], cmap=plt.cm.binary)
    plt.xlabel([quickdraw_train_y[i]])
plt.show()

# 模型加载
try:
    model = load_model(model_dir)
except IOError:
    print("model file doesn't exist! Need re-train!")
    # 构建Sequential模型层
    # 如下为VGG-19网络构建
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(3, 3), filters=32, strides=1, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=n_labels, activation='softmax')
    ])

    # 模型可视化
    model.summary()

    # 定义模型优化器
    optimizer = tf.keras.optimizers.Adadelta()
    # 定义损失函数
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    # 定义模型评估方法: 混淆矩阵
    accuracy = 'sparse_categorical_accuracy'

    # 模型编译
    model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])

    # 输入数据集，进行模型训练
    # history = model.fit(image_gen_train.flow(quickdraw_train_x, quickdraw_train_y, batch_size=64), epochs=epochs,
    #                     validation_data=(quickdraw_test_x, quickdraw_test_y), validation_freq=1, verbose=2, shuffle=True)
    history = model.fit(quickdraw_train_x, quickdraw_train_y, batch_size=64, epochs=epochs,
                        validation_data=(quickdraw_test_x, quickdraw_test_y), validation_freq=1, verbose=2,
                        shuffle=True)
    '''
    verbose：日志显示
    verbose = 0 不在标准输出流输出日志信息
    verbose = 1 输出进度条记录
    verbose = 2 每个epoch输出一行记录

    shuffle = true   是否在每轮迭代前，混洗数据
    '''

    model.summary()

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
model.evaluate(quickdraw_test_x, quickdraw_test_y, verbose=2)

# 模型持久化
model.save(model_dir)
