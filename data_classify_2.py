# coding=UTF-8
# 使用特征列对结构化数据进行分类
# 

'''
@File: data_classify_2.py
@Author: Wei Wei
@Time: 2022/9/11 0:14
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow import feature_column
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


batch_size = 32
epoch = 100
train_ds = df_to_dataset(train_data, batch_size=batch_size)
val_ds = df_to_dataset(val_data, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test_data, shuffle=False, batch_size=batch_size)

[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
print('A batch of ages:', train_features['age'])
print('A batch of label:', label_batch)

# 我们将使用该批数据演示几种特征列
example_batch = next(iter(train_ds))[0]
feature_columns = []


# 用于创建一个特征列
# 并转换一批次数据的一个实用程序方法
def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    # print(feature_layer(example_batch).numpy())


# Numeric feature 数值列
age = feature_column.numeric_column("age")
demo(age)
feature_columns.append(age)
fnlwgt = feature_column.numeric_column("fnlwgt")
demo(fnlwgt)
feature_columns.append(fnlwgt)
capital_gain = feature_column.numeric_column("capital_gain")
demo(capital_gain)
feature_columns.append(capital_gain)
capital_loss = feature_column.numeric_column("capital_loss")
demo(capital_loss)
feature_columns.append(capital_loss)
hours_per_week = feature_column.numeric_column("hours_per_week")
demo(hours_per_week)
feature_columns.append(hours_per_week)

# bucketized feature 分桶列
education_years = feature_column.numeric_column("education_years")
boundaries = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
education_years_buckets = feature_column.bucketized_column(education_years, boundaries=boundaries)
demo(education_years_buckets)
feature_columns.append(education_years_buckets)

# categorical feature 分类列
workclass_cate = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov',
                  'State-gov', 'Without-pay', 'Never-worked']
workclass = feature_column.categorical_column_with_vocabulary_list('workclass', workclass_cate)
workclass_one_hot = feature_column.indicator_column(workclass)
demo(workclass_one_hot)
feature_columns.append(workclass_one_hot)

education_cate = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm',
                  'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th',
                  'Doctorate', '5th-6th', 'Preschool']
education = feature_column.categorical_column_with_vocabulary_list('education', education_cate)
education_one_hot = feature_column.indicator_column(education)
demo(education_one_hot)
feature_columns.append(education_one_hot)

marital_cate = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                'Married-spouse-absent', 'Married-AF-spouse']
marital = feature_column.categorical_column_with_vocabulary_list('marital', marital_cate)
marital_one_hot = feature_column.indicator_column(marital)
demo(marital_one_hot)
feature_columns.append(marital_one_hot)

occupation_cate = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                   'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
                   'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                   'Armed-Forces']
occupation = feature_column.categorical_column_with_vocabulary_list('occupation', occupation_cate)
occupation_one_hot = feature_column.indicator_column(occupation)
demo(occupation_one_hot)
feature_columns.append(occupation_one_hot)

relationship_cate = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
relationship = feature_column.categorical_column_with_vocabulary_list('relationship', relationship_cate)
relationship_one_hot = feature_column.indicator_column(relationship)
demo(relationship_one_hot)
feature_columns.append(relationship_one_hot)

race_cate = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
race = feature_column.categorical_column_with_vocabulary_list('race', race_cate)
race_one_hot = feature_column.indicator_column(race)
demo(race_one_hot)
feature_columns.append(race_one_hot)

sex_cate = ['Female', 'Male']
sex = feature_column.categorical_column_with_vocabulary_list('sex', sex_cate)
sex_one_hot = feature_column.indicator_column(sex)
demo(sex_one_hot)
feature_columns.append(sex_one_hot)

native_country_cate = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                       'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China',
                       'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica',
                       'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic',
                       'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
                       'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador',
                       'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
native_country = feature_column.categorical_column_with_vocabulary_list('native_country', native_country_cate)
native_country_one_hot = feature_column.indicator_column(native_country)
demo(native_country_one_hot)
feature_columns.append(native_country_one_hot)

# 创建模型
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dropout(.1),
    layers.Dense(1)
])

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
# model.summary()
# 可视化模型
# rankdir='LR' is used to make the graph horizontal.
model_dir = project_path.rootPath + '/model/adult_income_2'
# tf.keras.utils.plot_model(model, to_file=model_dir + '/model_graph.png', show_shapes=True, rankdir="LR")

# 训练模型
history = model.fit(train_ds, validation_data=val_ds, epochs=epoch)

# 显示训练集和验证集的acc和loss曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 模型测试
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
