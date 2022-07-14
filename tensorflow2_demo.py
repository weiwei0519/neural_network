# coding=UTF-8
# tensorflow2 demo

'''
File Name: tensorflow2_demo.py
Program IDE: PyCharm
Created Time: 2022/6/26 0026 10:11
Author: Wei Wei
'''

import tensorflow as tf
import tensorflow_datasets as tfds

gpus = tf.config.list_physical_devices(device_type='GPU')
cpus = tf.config.list_physical_devices(device_type='CPU')
print(gpus, cpus)
# 查询tensorflow的数据集
tfds.list_builders()
#
# t1 = tf.Variable(42)
# t2 = tf.Variable([[[0., 1., 2.], [3., 4., 5.]], [[6., 7., 8.], [9., 10., 11]]])
# print(t1)
# print(t2)
# print(t2.shape)
#
# f64 = tf.Variable(89, dtype=tf.float64)
# print(f64.dtype)
#
# c1 = tf.constant(42)
# c2 = tf.constant(1, dtype=tf.int64)
# print(c1)
# print(c2)
#
# r1 = tf.reshape(t2, [2, 6])
# r2 = tf.reshape(t2, [1, 12])
# print(r1)
# print(r2)
#
# # 张量的秩
# rk1 = tf.rank(t2)
# rk2 = tf.rank(r1)
# print(rk1)
# print(rk2)
#
# t3 = t2[1, 0, 2]
# print(t3)
#
# # 张量转化为numpy
# n1 = t2.numpy()
# print(n1)
#
# # 计算张量的大小（元素数）
# si = tf.size(input=t2).numpy()
# print(si)
#
# # 张量的运算
# t4 = t2 * t2
# t5 = t2 * 4
# print(t4)
# print(t5)
#
# # tensor的转置与矩阵乘法
# u = tf.constant([[3, 4, 3]])
# v = tf.constant([[1, 2, 1]])
# w = tf.matmul(u, tf.transpose(v))
# print(w)
#
# # 张量数据类型的转换
# i = tf.cast(t1, dtype=tf.int32)
# j = tf.cast(tf.constant(4.9), dtype=tf.int32)
# print(i)
# print(j)
#
# # 计算张量的平方差
# x = tf.Variable([1, 3, 5, 7, 9])
# y = tf.Variable(5)
# mean_sqr = tf.math.squared_difference(x, y)
# print(mean_sqr)
#
# # 计算各个维度轴的平均值
# num = tf.constant([[4., 5.], [7., 3.]])
# all_avg = tf.reduce_mean(num, axis=None)  # 计算跨所有维度轴的平均值
# axis0_avg = tf.reduce_mean(num, axis=0)
# axis1_avg = tf.reduce_mean(num, axis=1)
# print(all_avg)
# print(axis0_avg)
# print(axis1_avg)
#
# # 张量随机初始化
# # 正态分布随机数
# tn = tf.random.normal(shape=(3, 2), mean=10, stddev=2, dtype=tf.float32, seed=None, name=None)
# # 均匀分布随机数
# tu = tf.random.uniform(shape=(3, 2), minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
# print(tn, tu)
#
# dice1 = tf.Variable(tf.random.uniform([10, 1], minval=1, maxval=7, dtype=tf.float32))
# dice2 = tf.Variable(tf.random.normal([10, 1], mean=10, stddev=2, dtype=tf.float32))
# res_mat = tf.concat(values=[dice1, dice2], axis=1)  # 将两个张量拼接
# print(dice1)
# print(dice2)
# print(res_mat)
#
# # 查找最小，最大元素的索引
# max = tf.argmax(res_mat, axis=None, name=None, output_type=tf.int32)  # 注意argmax，argmin返回的是索引，type只能为int32
# min = tf.argmin(res_mat, axis=0, name=None, output_type=tf.int32)
# print(max)
# print(min)
#
#
# def f1(x, y):
#     return tf.reduce_mean(input_tensor=tf.multiply(x ** 2, 5) + y ** 2)
#
# f2 = tf.function(f1)
# x = tf.constant([4., -5.])
# y = tf.constant([2., 3.])
# assert f1(x, y).numpy() == f2(x, y).numpy()
#

