# coding=UTF-8
# 使用tensorflow_datasets.MNIST数据集进行图像分类的神经网络模型

'''
File Name: mnist_classify.py
Program IDE: PyCharm
Created Time: 2022/6/21 0021 10:34
Author: Wei Wei
'''

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from util.pathutil import PathUtil

mnist = tf.keras.datasets.mnist
(mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = mnist.load_data()

epochs = 10  # 定义将数据输入模型的次数
batch_size = 32  # 定义每个批次输入数据量的大小

# 将数据集归一化为[0，1]，并转换数据类型
mnist_train_x = tf.cast(mnist_train_x / 255.0, dtype=tf.float32)
mnist_test_x = tf.cast(mnist_test_x / 255.0, dtype=tf.float32)
mnist_train_y = tf.cast(mnist_train_y, dtype=tf.int64)
mnist_test_y = tf.cast(mnist_test_y, dtype=tf.int64)

# show example 数据
plt.figure(figsize=(28, 28))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(mnist_train_x[i], cmap=plt.cm.binary)
    plt.xlabel([mnist_train_y[i]])
plt.show()

# 构建Sequential模型层
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 模型可视化
model.summary()

# 定义模型优化器
optimizer = tf.keras.optimizers.Adam()
# 定义损失函数
loss = tf.keras.losses.SparseCategoricalCrossentropy()
# 定义模型评估方法: 混淆矩阵
accuracy = 'accuracy'

# 模型编译
model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])

# 输入数据集，进行模型训练
model.fit(mnist_train_x, mnist_train_y, batch_size=batch_size, epochs=epochs, verbose=2)
'''
verbose：日志显示
verbose = 0 不在标准输出流输出日志信息
verbose = 1 输出进度条记录
verbose = 2 每个epoch输出一行记录
'''

# 模型评估
model.evaluate(mnist_test_x, mnist_test_y)

# 模型持久化
project_path = PathUtil()
model_dir = project_path.rootPath + '/model/mnist'
model.save(model_dir + '/mnist_classify.h5')

# 模型加载
new_model = load_model(model_dir + '/mnist_classify.h5')

