# coding=UTF-8
# 使用深度前馈神经网络实现XOR非线性分类模型

'''
File Name: xor_test.py
Program IDE: PyCharm
Created Time: 2022/6/16 0016 13:27
Author: Wei Wei
'''

from feedforward.Feedforward_NeuralNetwork import Feedforward_NeuralNetwork
import numpy as np
'''
利用简单非线性关系的数据集测试Implement
'''
if __name__ == '__main__':
    nn = Feedforward_NeuralNetwork([2, 2, 1], 'tanh')  # 输入层2个神经元， 隐藏层2个神经元， 输出层1个神经单元
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    nn.fit(X, Y)
    for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(i, nn.predict(i))