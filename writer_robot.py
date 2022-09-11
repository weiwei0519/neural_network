# coding=UTF-8
# 使用RNN（循环神经网络）训练写作模型
# 训练数据集为一篇小说

'''
@File: writer_robot.py
@Author: Wei Wei
@Time: 2022/7/17 15:18
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import tensorflow as tf
from keras.layers import CuDNNGRU
import numpy as np
import os
import time
from util.pathutil import PathUtil
import functools

# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 默认设置，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# 判断GRU模型，当前环境是否支持GPU
if tf.test.is_gpu_available():
    recurrent_nn = CuDNNGRU
    print('Current training based on GPU!')
else:
    recurrent_nn = functools.partial(tf.keras.layers.GRU, recurrent_activation='sigmoid')
    print('Current training based on CPU!')

file = '1400-0.txt'
url = 'https://www.gutenberg.org/files/1400/1400-0.txt'
tf.random.set_seed(2345)
project_path = PathUtil()
model_dir = project_path.rootPath + '/model/writer_RNN'
path = tf.keras.utils.get_file(file, url)
text = open(path, encoding='utf-8').read()
print('Length of text: {0} characters'.format(len(text)))
directory = model_dir + '/checkpoints'

# 去掉不参与训练的文本
text = text[1794:]
# 文本preview
print(text[:300])

# 获取文本中的非重复字符
vocabulary = sorted(set(text))
print('{0} unique characters.'.format(len(vocabulary)))

# 创建非重复字符（键）- 索引字典
char_to_index = {char: index for index, char in enumerate(vocabulary)}
print(char_to_index)
index_to_char = np.array(vocabulary)
print(index_to_char)

text_as_int = np.array([char_to_index[char] for char in text])
print('{')
for char, _ in zip(char_to_index, range(len(vocabulary))):
    print('{:4s}:{:3d}'.format(repr(char), char_to_index[char]))
print('\n}')

# 单个输入句子的的最大长度
sequence_length = 100
examples_per_epoch = len(text)

# 创建训练样本
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(sequence_length + 1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(1):
    print('Input data: ', repr(''.join(index_to_char[input_example.numpy()])))
    print('Target data:', repr(''.join(index_to_char[target_example.numpy()])))

for char, (input_index, target_index) in enumerate(zip(input_example[:5], target_example[:5])):
    print('Step {:4d}'.format(char))
    print(' input: {} ({:s})'.format(input_index, repr(index_to_char[target_index])))
    print(' expeted output: {} ({:s})'.format(target_index, repr(index_to_char[target_index])))

batch_size = 64
# 每轮的训练步数
steps_per_epoch = examples_per_epoch // batch_size
# tf数据在内存中有一个缓冲去，用于混排数据，定义buffer
buffer = 10000

dataset = dataset.shuffle(buffer).batch(batch_size, drop_remainder=True)
# 对数据集调用repeat(), 以便数据可以重复输入模型
dataset = dataset.repeat()

vocabulary_length = len(vocabulary)
# 嵌入维数
embedding_dimension = 256
# RNN神经元的个数
recurrent_nn_units = 1024

# 模型加载
try:
    # 重建模型
    rnn_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocabulary_length, embedding_dimension, batch_input_shape=[1, None]),
        CuDNNGRU(recurrent_nn_units, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True),
        tf.keras.layers.Dense(vocabulary_length)
    ])
    rnn_model.load_weights(tf.train.latest_checkpoint(directory))
    rnn_model.build(tf.TensorShape([1, None]))
    rnn_model.summary()
except IOError:
    print("model file doesn't exist! Need re-train!")
    # 构建模型
    rnn_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocabulary_length, embedding_dimension, batch_input_shape=[batch_size, None]),
        CuDNNGRU(recurrent_nn_units, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True),
        tf.keras.layers.Dense(vocabulary_length)
    ])

    rnn_model.summary()

    # 校验模型
    for batch_input_example, batch_target_example in dataset.take(1):
        batch_predictions_example = rnn_model(batch_input_example)
        print(batch_predictions_example.shape, "# (batch, sequence_length, vocabulary_length)")
    sample_indices = tf.random.categorical(logits=batch_predictions_example[0], num_samples=1)
    sample_indices = tf.squeeze(sample_indices, axis=-1).numpy()
    print(sample_indices)

    # 模型训练之前，查看输入和输出
    print("Input: \n", repr("".join(index_to_char[batch_input_example[0]])))
    print("Next Char predictions: \n", repr("".join(index_to_char[sample_indices])))

    # 定义损失函数
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    batch_loss_example = tf.compat.v1.losses.sparse_softmax_cross_entropy(batch_target_example,
                                                                          batch_predictions_example)
    rnn_model.compile(optimizer=tf.optimizers.Adam(), loss=loss)

    # 检查点文件
    file_prefix = os.path.join(directory, 'ckpt_{epoch}')
    callback = [tf.keras.callbacks.ModelCheckpoint(filepath=file_prefix, save_weights_only=True)]

    epochs = 45
    history = rnn_model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callback)

    # 检查最近一次的检查点
    tf.train.latest_checkpoint(directory)


def generate_text(rnn_model, start_string, temperature, characters_to_generate):
    # 将起始字符串向量化
    input_string = [char_to_index[char] for char in start_string]
    input_string = tf.expand_dims(input_string, 0)
    # 初始化空列表
    generated = []
    rnn_model.reset_states()
    for i in range(characters_to_generate):
        predictions = rnn_model(input_string)
        predictions = tf.squeeze(predictions, 0)
        # 使用随机分类（多项式）分布预测模型返回的字符
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(logits=predictions, num_samples=1)[-1, 0].numpy()
        # 将预测字符与上一个隐藏状态，共同作为下一个输入传递给模型
        input_string = tf.expand_dims([predicted_id], 0)
        generated.append(index_to_char[predicted_id])
    return (start_string + ''.join(generated))

generated_text = generate_text(rnn_model=rnn_model, start_string='Pip', temperature=0.1, characters_to_generate=1000)
print(generated_text)