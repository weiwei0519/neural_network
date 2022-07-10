# coding=UTF-8
# 基于tensorflow的深度前馈神经网络的实现
# 使用Cifar10数据集进行图像分类的全连接神经网络模型

'''
File Name: FFNN.py
Program IDE: PyCharm
Created Time: 2022/6/18 0018 10:41
Author: Wei Wei
'''

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from util.pathutil import PathUtil


def preprocess(images, labels):
    scale_layer = tf.keras.layers.Rescaling(1. / 255)
    images = scale_layer(images)
    return images, labels


class FFNN():

    def __init__(self, image_dir=None, labels=None, pixel=(32, 32)):
        self.labels = labels
        self.pixel = pixel
        self.image = None
        self.label = None
        self.train_image_ds, self.val_image_ds, self.label_class = self.load_image_ds(image_dir, pixel)
        print("initial FFNN done!")
        try:
            print("pre-load model file!")
            self.model = keras.models.load_model('./model')
        except OSError:
            print("model file is not exist. need re-train")
            self.model = self.model_train()

    def load_image_ds(self, image_dir, pixel):
        self.batch_size = 32
        train_image_ds = tf.keras.preprocessing.image_dataset_from_directory(image_dir,
                                                                             image_size=pixel,
                                                                             validation_split=0.3,
                                                                             label_mode='int',
                                                                             subset="training",
                                                                             batch_size=self.batch_size,
                                                                             seed=1,
                                                                             color_mode='grayscale',
                                                                             shuffle=True)
        val_image_ds = tf.keras.preprocessing.image_dataset_from_directory(image_dir,
                                                                           image_size=pixel,
                                                                           validation_split=0.3,
                                                                           label_mode='int',
                                                                           subset="validation",
                                                                           batch_size=self.batch_size,
                                                                           seed=1,
                                                                           color_mode='grayscale',
                                                                           shuffle=True)
        train_image_ds = train_image_ds.map(preprocess)
        val_image_ds = val_image_ds.map(preprocess)
        # for element in train_image_ds:
        #     print(element)
        for images, labels in train_image_ds.take(1):
            label_class = len(labels)

        return train_image_ds, val_image_ds, label_class

    def model_train(self):
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(input_shape=IMG_SHAPE, kernel_size=(5, 5), filters=32, activation='relu'),
        #     tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(256, activation='relu'),
        #     tf.keras.layers.Dropout(0.25),
        #     tf.keras.layers.Dense(units=n_classes, activation='softmax')
        # ])
        # Step1: 构建模型层
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(32, 32, 1)),  # 在数据输入进全连接层时，要对数据进行铺平处理
            keras.layers.Dense(1024, activation='relu'),  # 激活函数为整流线性单元 g(z) = max{0, z}, z = Wx + b
            keras.layers.Dropout(0.2),
            keras.layers.Dense(units=self.label_class)])

        # Step2: 编译模型参数
        # 损失函数loss - 用于测量模型在训练期间的准确率。
        # 优化器 - 决定模型如何根据其看到的数据和自身的损失函数进行更新。
        # 指标 - 用于监控训练和测试步骤。
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # Step3: 训练模型
        # epochs 迭代次数
        model.fit(self.train_image_ds, epochs=50, batch_size=32, validation_data=self.val_image_ds, verbose=2)

        # Step4: 保存训练好的模型
        model.save('./model', save_format='tf')
        return model

    def model_evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.validate_image, verbose=2)


if __name__ == '__main__':
    project_path = PathUtil()
    image_dir = project_path.rootPath + '/datasets/cifar10_CNN'
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    pixel = (32, 32)
    nn_model = FFNN(image_dir, labels, pixel)
