# coding=UTF-8
# ImageNet图像分类，基于tensorflow的CNN实现

import tensorflow as tf
import tensorflow_datasets as tfds
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from util.pathutil import PathUtil

# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 默认设置，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)
project_path = PathUtil()
model_dir = project_path.rootPath + '/model/ImageNet'

# 加载ImageNet数据集
ds = tfds.load('imagenet2012', split=['train[:80%]', 'test[:20%]'], shuffle_files=True,
               data_dir='C:\\Users\\WeiWei\\Development\\PyCharm_Project\\datasets\\ImageNet')
