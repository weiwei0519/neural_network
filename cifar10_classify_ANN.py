# coding=UTF-8
# cifar10进行图像分类的神经网络模型

'''
File Name: cifar10_CNN.py
Program IDE: PyCharm
Created Time: 2022/6/21 0021 11:20
Author: Wei Wei
'''

from util.pathutil import PathUtil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model
import os
from sklearn.model_selection import train_test_split


def save_datasets(dataset_dir, image_ds, label_ds):
    dataset_dir = dataset_dir + '\\datasets.pkl'
    dataset_file = open(dataset_dir, 'wb')
    pickle.dump(image_ds, dataset_file)
    pickle.dump(label_ds, dataset_file)


def load_datasets_from_file(dataset_dir):
    dataset_dir = dataset_dir + '/datasets.pkl'
    try:
        dataset_file = open(dataset_dir, 'rb')
        image_ds = pickle.load(dataset_file)
        label_ds = pickle.load(dataset_file)
        return image_ds, label_ds
    except FileNotFoundError:
        print("image datasets file doesn't exist. Need reload.")


# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 默认设置，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)
project_path = PathUtil()
image_dir = project_path.rootPath + '/datasets/cifar10'
pixel = (32, 32)
model_dir = project_path.rootPath + '/model/cifar10_ANN'
epochs = 20  # 定义将数据输入模型的次数
batch_size = 64  # 定义每个批次输入数据量的大小
n_labels = 10  # 分类集大小

# 根据图片路径，构造数据集，图片采用灰度。
image_ds, label_ds = load_datasets_from_file(image_dir)
if image_ds.any() == None:
    paths = list(image_dir.glob('*/*.png'))  # 图片全路径
    image_ds = np.zeros((len(paths), pixel[0], pixel[1]))
    label_ds = np.zeros((len(paths)))
    for i, path in enumerate(paths):
        if i % 1000 == 0:
            print("read No{0}".format(i))
        path = str(path)
        pl = path.split('\\')
        # 提取图像分类label
        label_ds[i] = pl[-2]
        # 读取图像，并转换为灰度
        image = tf.io.read_file(path)
        # image decode方法的channel参数的解释：0：使用JPEG编码图像中的通道数量（默认）；1：输出灰度图像；3：输出RGB图像.
        if pl[-1].find('jpeg') > 0:
            image = tf.image.decode_jpeg(image, channels=1)
        elif pl[-1].find('png') > 0:
            image = tf.image.decode_png(image, channels=1)
        elif pl[-1].find('bmp') > 0:
            image = tf.image.decode_bmp(image, channels=1)
        image = tf.squeeze(image, [2])
        image_ds[i] = image
    # 保存数据集
    save_datasets(image_dir, image_ds, label_ds)

# 训练集和测试集的拆分
cifar10_train_x, cifar10_test_x, cifar10_train_y, cifar10_test_y = train_test_split(image_ds, label_ds,
                                                                                    test_size=0.2,
                                                                                    random_state=42)

# 图片数据集预处理，将数据集归一化为[0，1]，并转换数据类型
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

# 模型加载
try:
    model = load_model(model_dir)
except IOError:
    print("model file doesn't exist! Need re-train!")
    # 构建Sequential模型层
    # 如下为基于ANN构建模型
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32)),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
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
