# coding=UTF-8
# 使用深度前馈神经网络实现手写识别

'''
File Name: HandwrittenDigitsRecognize.py.py
Program IDE: PyCharm
Created Time: 2022/6/16 0016 13:21
Author: Wei Wei
'''

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from feedforward.Feedforward_NeuralNetwork import Feedforward_NeuralNetwork
from sklearn.model_selection import train_test_split
import pylab as pl

'''
手写数字识别
'''

digits = load_digits()
X = digits.data  # 特征量
y = digits.target  # 标签
X -= X.min()  # 把所有的数字转化成0-1的大小，即标准化操作
X /= X.max()

print(digits.data.shape)

pl.gray()
pl.matshow(digits.images[0])
pl.show()

nn = Feedforward_NeuralNetwork([64, 100, 10], 'sigmoid')  # 每个图有64个像素点，10个要识别的数字
X_train, X_test, y_train, y_test = train_test_split(X, y)
labels_train = LabelBinarizer().fit_transform(y_train)  # 标签二值化
labels_test = LabelBinarizer().fit_transform(y_test)  # 标签二值化
print('start fitting----')
nn.fit(X_train, labels_train, epochs=3000)
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))  # 取概率最大的那个数字
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
