# coding=UTF-8
# 使用tensorflow对adult_income数据集进行分类
# 

'''
@File: income_classify.py
@Author: Wei Wei
@Time: 2022/9/10 16:04
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from util.pathutil import PathUtil
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 设置环境目录上下文
project_path = PathUtil()
data_dir = project_path.rootPath + '/datasets/adult_income'
model_dir = project_path.rootPath + '/model/adult_income'

# 加载训练集
pickle_file_X = open(data_dir + '/trainsets_X.pkl', 'rb+')
pickle_file_Y = open(data_dir + '/trainsets_Y.pkl', 'rb+')
X_train = pickle.load(pickle_file_X)
y_train = pickle.load(pickle_file_Y)
pickle_file_X.close()
pickle_file_Y.close()

# 加载测试集
pickle_file_X = open(data_dir + '/testsets_X.pkl', 'rb+')
pickle_file_Y = open(data_dir + '/testsets_Y.pkl', 'rb+')
X_validate = pickle.load(pickle_file_X)
y_validate = pickle.load(pickle_file_Y)
pickle_file_X.close()
pickle_file_Y.close()

# 归一化
def normalize(train, test):
    mean, std = train.mean(), test.std()
    train = (train - mean) / std
    test = (test - mean) / std
    return train, test

def normalize2(train, test):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.astype(np.float32).reshape(-1, 1)).reshape(1, -1)
    test_scaled = scaler.transform(test.astype(np.float32).reshape(-1, 1)).reshape(1, -1)
    return train_scaled, test_scaled

# 数据预处理
X_train_norm = np.zeros(X_train.shape)
X_val_norm = np.zeros(X_validate.shape)
for i in range(X_train.shape[1]):
    X_train_norm[:, i], X_val_norm[:, i] = normalize2(X_train[:, i], X_validate[:, i])
X_train = X_train_norm
X_validate = X_val_norm

print("train datasets X is maxtrix {0} x {1}".format(X_train.shape[0], X_train.shape[1]))
print(X_train)
print("train datasets Y is maxtrix {0} x {1}".format(y_train.shape[0], y_train.shape[1]))
print(y_train)

print("test datasets X is maxtrix {0} x {1}".format(X_validate.shape[0], X_validate.shape[1]))
print(X_validate)
print("test datasets Y is maxtrix {0} x {1}".format(y_validate.shape[0], y_validate.shape[1]))
print(y_validate)

# 训练集和测试集的拆分
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

epochs = 20  # 定义将数据输入模型的次数
batch_size = 32  # 定义每个批次输入数据量的大小

# 构建Sequential模型层
model = tf.keras.models.Sequential([
    # tf.keras.layers.Flatten(input_shape=(14)),
    tf.keras.layers.Dense(14, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
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
# 输入数据集，进行模型训练
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_test, y_test), validation_freq=1, verbose=2)
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
model.evaluate(X_validate, y_validate)

# 模型持久化
project_path = PathUtil()
model.save(model_dir + '/adult_income_classify.h5')

# 模型加载
new_model = load_model(model_dir + '/adult_income_classify.h5')