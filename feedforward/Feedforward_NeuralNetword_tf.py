# coding=UTF-8
# 基于tensorflow的深度前馈神经网络的实现

'''
File Name: Feedforward_NeuralNetword_tf.py
Program IDE: PyCharm
Created Time: 2022/6/16 0016 20:13
Author: Wei Wei
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import datetime


# 定义一个想要拟合的函数, 先取了一个带高斯噪音的x_data, y_data，作为训练集。x_val_data, y_val_data, 作为验证集，
# 一般要把拥有的数据库分为三部分，训练集60%，验证集20%，测试集20%
def create_datasets():
    x_data = np.linspace(-1, 30, 3000)  # 生成等间距数组的的一个函数
    noise = np.random.normal(0, 0.05, x_data.shape)
    # y_data = 5 * np.exp(-x_data / 4.56) + noise
    y_data = np.sin(x_data) + noise + 5 * np.exp(-x_data / 4.56)
    x_val_data = np.linspace(10, 20, 1000)
    y_val_data = np.sin(x_val_data) + 5 * np.exp(-x_val_data / 4.56)
    return x_data, y_data, x_val_data, y_val_data


# 建立一个全连接的模型
def feedforward_neuralnetword_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(1,)),  # 在数据输入进全连接层时，要对数据进行铺平处理
            tf.keras.layers.Dense(300, activation='relu'),  # 激活函数为整流线性单元 g(z) = max{0, z}, z = Wx + b
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            # 设置dropout ，防止过拟合, 如果在误差稳定时训练误差小而验证误差很大，可能是出现了过拟合现象，
            # 可以加入dropout层或进行参数正则化等
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)  # 设置一个神经元，输出预测的y列表
        ]
    )
    optimizer = tf.keras.optimizers.Adam(0.01)
    model.compile(optimizer=optimizer, loss="mse")  # 优化目标为“mse”使均方差函数最小
    return model


if __name__ == '__main__':
    time1 = time.time()
    timex = []
    loss = []
    val_loss = []
    x_data, y_data, x_val_data, y_val_data = create_datasets()
    model = feedforward_neuralnetword_model()
    log_dir = 'E:\\Development\\logs\\Pycharm_Projects'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    summary_writer = tf.summary.create_file_writer(log_dir, )

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    bx = fig.add_subplot(2, 2, 2)
    cx = fig.add_subplot(2, 2, 3)
    dx = fig.add_subplot(2, 2, 4)
    ax.plot(x_data, y_data)
    cx.plot(x_val_data, y_val_data)
    plt.ion()
    plt.show()

    for i in range(2000):

        hist = model.fit(x_data, y_data, batch_size=256, epochs=5, validation_data=(x_val_data, y_val_data))
        timex.append(time.time() - time1)
        loss.append(hist.history['loss'])
        val_loss.append(hist.history['val_loss'])
        print(hist.history['loss'])
        with summary_writer.as_default():
            tf.summary.scalar('train-loss', float(hist.history['loss'][0]), step=i)
            tf.summary.scalar('val-loss', float(hist.history['val_loss'][0]), step=i)
        if i % 2 == 0:
            try:
                ax.lines.remove(lines[0])
                bx.lines.remove(lines2[0])
                cx.lines.remove(lines4[0])
                dx.lines.remove(lines3[0])
            except Exception:
                pass

            y_pred = model.predict(x_data)
            lines4 = cx.plot(x_val_data, model.predict(x_val_data))
            lines3 = dx.plot(timex, val_loss)
            lines2 = bx.plot(timex, loss)
            lines = ax.plot(x_data, y_pred)
            plt.pause(0.0001)
        if tf.reduce_max(hist.history['val_loss']) < 0.0003:
            # model.save(filepath="D:\\requests\\keras_network_savedModel")
            break

plt.pause(10)
