# coding=UTF-8
# 使用ANN对图像进行压缩编码和解压编码

'''
File Name: auto_encoder.py
Program IDE: PyCharm
Created Time: 2022/7/4 0004 21:06
Author: Wei Wei
'''

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
from util.pathutil import PathUtil
import os
import numpy as np

# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 默认设置，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)
project_path = PathUtil()
model_dir = project_path.rootPath + '/model/auto_encoder'

# 加载数据集
fashion_mnist = tf.keras.datasets.fashion_mnist
(fashion_mnist_train_x, _), (fashion_mnist_test_x, _) = fashion_mnist.load_data()

pixel = (28, 28)  # 图片像素
epochs = 50  # 定义将数据输入模型的次数
batch_size = 64  # 定义每个批次输入数据量的大小
org_img_dim = pixel[0] * pixel[1]  # 输入图像的维度
zip_img_dim = 32  # 编码压缩后的图像维度，压缩比：784 / 32 = 24.5

# 将数据集归一化为[0，1]，并转换数据类型
fashion_mnist_train_x = fashion_mnist_train_x.astype('float32') / 255.
fashion_mnist_test_x = fashion_mnist_test_x.astype('float32') / 255.

# show example 数据
plt.figure(figsize=pixel)
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(fashion_mnist_train_x[i], cmap=plt.cm.binary)
    plt.xlabel([])
plt.show()

# 模型加载
try:
    model = load_model(model_dir)
except IOError:
    print("model file doesn't exist! Need re-train!")
    # 构建Sequential模型层
    # 将图像输入一个一维的全连接层，需要flatten
    fashion_mnist_train_x = fashion_mnist_train_x.reshape((fashion_mnist_train_x.shape[0],
                                                           np.prod(fashion_mnist_train_x.shape[1:])))
    fashion_mnist_test_x = fashion_mnist_test_x.reshape((fashion_mnist_test_x.shape[0],
                                                         np.prod(fashion_mnist_test_x.shape[1:])))
    # 如下为基于ANN
    # 创建单层编码器，以及自动编码的训练模型（有单层编码器 + 单层解码器组成）
    input_image = Input(shape=(org_img_dim,))
    encoded_image = Dense(zip_img_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_image)
    encoder = Model(input_image, encoded_image)  # 编码模型
    decoded_image = Dense(org_img_dim, activation='sigmoid')(encoded_image)
    autoencoder = Model(input_image, decoded_image)  # 从编码输入，到解码输出的整体模型，需要针对此模型进行训练，
    # 模型将原始图片的输入，映射到新图片（像素一样）输出的模型
    # 创建单层解码器
    encoded_input = Input(shape=(zip_img_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    # 模型可视化
    autoencoder.summary()

    # 定义模型优化器
    optimizer = 'adadelta'
    # 定义损失函数
    loss = 'binary_crossentropy'
    # 定义模型评估方法: 混淆矩阵
    accuracy = ''

    # 编译自动编码器
    autoencoder.compile(optimizer=optimizer, loss=loss)

    # 输入数据集，进行模型训练
    history = autoencoder.fit(fashion_mnist_train_x, fashion_mnist_train_x,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(fashion_mnist_test_x, fashion_mnist_test_x),
                              validation_freq=1,
                              verbose=2,
                              shuffle=True)
    '''
    verbose：日志显示
    verbose = 0 不在标准输出流输出日志信息
    verbose = 1 输出进度条记录
    verbose = 2 每个epoch输出一行记录
    '''

    # 显示训练集和验证集的acc和loss曲线
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # 模型持久化
    autoencoder.save(model_dir)

    # 模型评估
    encoded_imgs = encoder.predict(fashion_mnist_test_x)  # 压缩编码
    edcoded_imgs = decoder.predict(encoded_imgs)  # 解压编码

    # 压缩前后的图片显示
    num_of_images = 12
    plt.figure(figsize=(20, 4))
    for i in range(num_of_images):
        # 压缩前的图片显示
        graph = plt.subplot(2, num_of_images, i + 1)
        plt.imshow(fashion_mnist_test_x[i].reshape(pixel[0], pixel[1]))
        plt.gray()
        graph.get_xaxis().set_visible(False)
        graph.get_yaxis().set_visible(False)
        # 解压缩后的显示
        graph = plt.subplot(2, num_of_images, i + 1 + num_of_images)
        plt.imshow(edcoded_imgs[i].reshape(pixel[0], pixel[1]))
        plt.gray()
        graph.get_xaxis().set_visible(False)
        graph.get_yaxis().set_visible(False)
    plt.show
