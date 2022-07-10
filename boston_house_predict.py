# coding=UTF-8
# 基于tensorflow的线性回归模型
# 用于Boston房价数据预测

'''
File Name: boston_house_predict.py
Program IDE: PyCharm
Created Time: 2022/7/2 0002 9:36
Author: Wei Wei
'''

import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from util.pathutil import PathUtil
from sklearn.preprocessing import scale

# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 默认设置，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)
project_path = PathUtil()
model_dir = project_path.rootPath + '/model/boston_house_predict'

# 加载Boston房价数据集
boston_housing = tf.keras.datasets.boston_housing
(boston_housing_train_x, boston_housing_train_y), (
    boston_housing_test_x, boston_housing_test_y) = boston_housing.load_data()

# 将数据集归一化为均值为0，标准差为1的数据集
boston_housing_train_x = tf.cast(scale(boston_housing_train_x), dtype=tf.float32)
boston_housing_test_x = tf.cast(scale(boston_housing_test_x), dtype=tf.float32)
boston_housing_train_y = tf.cast(boston_housing_train_y, dtype=tf.float32)
boston_housing_test_y = tf.cast(boston_housing_test_y, dtype=tf.float32)

epochs = 1000  # 定义将数据输入模型的次数
learning_rate = 0.1


# 预测函数，采用线性回归模型
def prediction(x, weight, bias):
    return tf.matmul(x, weight) + bias


# 损失函数使用均方误差
def loss(x, y, weight, bias):
    pred_y = tf.squeeze(prediction(x, weight, bias))
    error = pred_y - y
    squared_error = tf.square(error)
    return tf.sqrt(tf.reduce_mean(input_tensor=squared_error))


# 损失梯度计算
def gradient(x, y, weight, bias):
    with tf.GradientTape() as tape:
        tape.watch([weight, bias])
        loss_value = loss(x, y, weight, bias)
        delatW, deltaB = tape.gradient(loss_value, [weight, bias])
    return delatW, deltaB, loss_value


W = tf.random.normal([13, 1], mean=0.0, stddev=1.0, dtype=tf.float32)
B = tf.zeros(1, dtype=tf.float32)
# B = tf.Variable(tf.zeros(1), dtype=tf.float32)
print(W, B)
print("Initial loss: {:.3f}".format(loss(boston_housing_train_x, boston_housing_train_y, W, B)))

train_loss = []
test_loss = []

for e in range(epochs):
    deltaW, deltaB, train_loss_value = gradient(boston_housing_train_x, boston_housing_train_y, W, B)
    change_W = deltaW * learning_rate
    change_B = deltaB * learning_rate
    W = W - change_W  # W = W - change_W
    B = B - change_B  # B = B - change_B
    test_loss_value = loss(boston_housing_test_x, boston_housing_test_y, W, B)
    train_loss.append(train_loss_value)
    test_loss.append(test_loss_value)
    print("Training loss and Validation loss after epoch {:02d}: [{:.3f}, {:.3f}]".format(e, train_loss_value,
                                                                                          test_loss_value))

# 显示训练集和验证集的loss曲线
plt.figure(figsize=(8, 8))
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()