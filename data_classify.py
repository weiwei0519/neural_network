# coding=UTF-8
# 使用 Keras 预处理层对结构化数据进行分类
# 

'''
@File: data_classify.py
@Author: Wei Wei
@Time: 2022/9/10 17:01
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import pathlib
from util.pathutil import PathUtil
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental import preprocessing
from keras.utils.vis_utils import plot_model

# 设置环境目录上下文
project_path = PathUtil()
# 加载训练集
train_data_path = project_path.rootPath + '/datasets/adult_income/income_trainsets.csv'
train_data = pd.read_csv(train_data_path)
# data = data.sample(1000)
train_data.info()
# 加载测试集
test_data_path = project_path.rootPath + '/datasets/adult_income/income_testsets.csv'
test_data = pd.read_csv(test_data_path)
test_data.info()

# 转化label值
train_data['label'] = np.where(train_data['income'] == ' <=50K', 0, 1)
train_data = train_data.drop(columns=['income'])
test_data['label'] = np.where(test_data['income'] == ' <=50K', 0, 1)
test_data = test_data.drop(columns=['income'])

# 从训练集中拆分验证集，20%
train_data, val_data = train_test_split(train_data, test_size=0.2)

print(len(train_data), 'train examples')
print(len(val_data), 'validation examples')
print(len(test_data), 'test examples')


# 使用 tf.data 创建输入流水线
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('label')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


batch_size = 256
epoch = 100
train_ds = df_to_dataset(train_data, batch_size=batch_size)
val_ds = df_to_dataset(val_data, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test_data, shuffle=False, batch_size=batch_size)

[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
print('A batch of ages:', train_features['age'])
print('A batch of label:', label_batch)


# 数值列
# 对于每个数值特征，使用 Normalization() 层来确保每个特征的平均值为 0，且其标准差为 1。
# get_normalization_layer 函数返回一个层，该层将特征归一化应用于数值特征。
def get_normalization_layer(name, dataset):
    # Create a Normalization layer for our feature.
    normalizer = preprocessing.Normalization(axis=None)
    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(lambda x, y: x[name])
    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)
    return normalizer


# 分类列
# 在此数据集中，Type 表示为字符串（例如 ' Married-civ-spouse' 或 ' Never-married'）。
# 您不能将字符串直接馈送给模型。预处理层负责将字符串表示为独热向量。
# get_category_encoding_layer 函数返回一个层，该层将值从词汇表映射到整数索引，并对特征进行独热编码。
def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    # Create a StringLookup layer which will turn strings into integer indices
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_tokens=max_tokens)
    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)
    # Create a Discretion for our integer indices.
    encoder = preprocessing.CategoryEncoding(num_tokens=index.vocabulary_size())
    # Apply one-hot encoding to our indices. The lambda function captures the
    # layer, so we can use them, or include them in the functional model later.
    return lambda feature: encoder(index(feature))


all_inputs = []
encoded_features = []

# Numeric features. 数值类型的feature
for header in ['fnlwgt', 'capital_gain', 'capital_loss']:
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)

# Categorical features encoded as integers. 分类类型的feature，数值类型，转换为独热编码
categorical_int_cols = ['age', 'education_years', 'hours_per_week']
for header in categorical_int_cols:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='int64')
    encoding_layer = get_category_encoding_layer(header, train_ds, dtype='int64',
                                                 max_tokens=5)
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)

# Categorical features encoded as string. 分类类型的feature，字符类型，转换为独热编码
categorical_string_cols = ['workclass', 'education', 'marital', 'occupation', 'relationship', 'race', 'sex',
                           'native_country']
for header in categorical_string_cols:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string',
                                                 max_tokens=5)
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)

# 创建模型
all_features = tf.keras.layers.concatenate(encoded_features)
x1 = tf.keras.layers.Dense(128, activation="relu")(all_features)
x2 = tf.keras.layers.Dropout(0.5)(x1)
output = tf.keras.layers.Dense(1)(x2)
model = tf.keras.Model(all_inputs, output)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])
# 可视化模型
# rankdir='LR' is used to make the graph horizontal.
model_dir = project_path.rootPath + '/model/adult_income'
tf.keras.utils.plot_model(model, to_file=model_dir+'/model_graph.png', show_shapes=True, rankdir="LR")

# 模型训练
history = model.fit(train_ds, epochs=50, validation_data=val_ds)

# 显示训练集和验证集的acc和loss曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# 模型预测
test_loss, test_accuracy = model.evaluate(test_ds)
print("Accuracy", test_accuracy)

# 模型保存
model.save(model_dir + '/adult_income_classifier')
reloaded_model = tf.keras.models.load_model(model_dir + '/adult_income_classifier')

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
plt.show(block=True)
