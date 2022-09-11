# coding=UTF-8
# 使用RNN（循环神经网络）训练写作模型
# 训练数据集为一篇小说

'''
@File: text_generator_RNN.py
@Author: Wei Wei
@Time: 2022/7/18 22:43
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import tensorflow as tf
import numpy as np
import os
from util.pathutil import PathUtil

# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 默认设置，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

file = '1400-0.txt'
url = 'https://www.gutenberg.org/files/1400/1400-0.txt'  # Great Expectations by Charles Dickens
path = tf.keras.utils.get_file(file, url)
text = open(path, encoding='utf-8').read()
print('Length of text: {} characters'.format(len(text)))
# strip off text we don't need
text = text[1794:]
# Take a look at the first 300 characters in text
print(text[:300])
# The unique characters in the file
vocabulary = sorted(set(text))
# print('{} unique characters.'.format(len(vocabulary)))

vocabulary_size = len(vocabulary)
# Creating a dictionary of unique characters to indices
char_to_index = {char: index for index, char in enumerate(vocabulary)}
# print(char_to_index)
index_to_char = np.array(vocabulary)
# print(index_to_char)
text_as_int = np.array([char_to_index[char] for char in text])
# print('{')
# for char, _ in zip(char_to_index, range(20)):
#     print('  {:4s}: {:3d},'.format(repr(char), char_to_index[char]))
# print('  ...\n}')
# # Show how the first 15 characters from the text are mapped to integers
# print('{} ---- characters mapped to int ---- > {}'.format(repr(text[:15]), text_as_int[:15]))

# The maximum length sentence we want for a single input in characters
sequence_length = 100
examples_per_epoch = len(text) // sequence_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for char in char_dataset.take(5):
    print(char.numpy())
    print(index_to_char[char.numpy()])

sequences = char_dataset.batch(sequence_length + 1, drop_remainder=True)
for item in sequences.take(5):
    print(repr(''.join(index_to_char[item.numpy()])))


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(1):
    print('Input data: ', repr(''.join(index_to_char[input_example.numpy()])))
    print('Target data:', repr(''.join(index_to_char[target_example.numpy()])))

    for char, (input_index, target_index) in enumerate(zip(input_example[:5], target_example[:5])):
        print("Step {:4d}".format(char))
        print("  input: {} ({:s})".format(input_index, repr(index_to_char[input_index])))
        print("  expected output: {} ({:s})".format(target_index, repr(index_to_char[target_index])))

# Batch size
batch = 64
steps_per_epoch = examples_per_epoch // batch

# TF data maintains a buffer in memory in which to shuffle data
# since it is designed to work with possibly endless data
buffer = 10000
# 用shuffle打乱dataset数据集的顺序，以防止过拟合
# 每epoch获取batch个数的数据集作为模型输入
dataset = dataset.shuffle(buffer).batch(batch, drop_remainder=True)

dataset = dataset.repeat()

#  The vocabulary length in characterrs
vocabulary_length = len(vocabulary)

# The embedding dimension
embedding_dimension = 256

# Number of RNN units
recurrent_nn_units = 1024

if tf.test.is_gpu_available():
    recurrent_nn = tf.compat.v1.keras.layers.CuDNNGRU
    print("GPU in use")
else:
    import functools

    recurrent_nn = functools.partial(tf.keras.layers.GRU, recurrent_activation='sigmoid')
    print("CPU in use")


def build_model(vocabulary_size, embedding_dimension, recurrent_nn_units, batch_size):
    model = tf.keras.Sequential(
        [tf.keras.layers.Embedding(vocabulary_size, embedding_dimension, batch_input_shape=[batch_size, None]),
         recurrent_nn(recurrent_nn_units, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True),
         tf.keras.layers.Dense(vocabulary_length)
         ])
    return model


model = build_model(
    vocabulary_size=len(vocabulary),
    embedding_dimension=embedding_dimension,
    recurrent_nn_units=recurrent_nn_units,
    batch_size=batch)

# 使用一个批次，验证模型构建的正确性
for batch_input_example, batch_target_example in dataset.take(1):
    batch_predictions_example = model(batch_input_example)
    print(batch_predictions_example.shape, "# (batch, sequence_length, vocabulary_length)")

model.summary()

sampled_indices = tf.random.categorical(logits=batch_predictions_example[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
print("Input: \n", repr("".join(index_to_char[batch_input_example[0]])))
print("Next Char Predictions: \n", repr("".join(index_to_char[sampled_indices])))


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


batch_loss_example = tf.compat.v1.losses.sparse_softmax_cross_entropy(batch_target_example, batch_predictions_example)
print("Prediction shape: ", batch_predictions_example.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", batch_loss_example.numpy())

model.compile(optimizer=tf.optimizers.Adam(), loss=loss)

# Directory where the checkpoints will be saved
project_path = PathUtil()
model_dir = project_path.rootPath + '/model/text_generator'
directory = model_dir + '/checkpoints'
# Name of the checkpoint files
file_prefix = os.path.join(directory, "ckpt_{epoch}")

callback = [tf.keras.callbacks.ModelCheckpoint(filepath=file_prefix, save_weights_only=True)]

epochs = 45

history = model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callback)

tf.train.latest_checkpoint(directory)

model = build_model(vocabulary_size, embedding_dimension, recurrent_nn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(directory))
model.build(tf.TensorShape([1, None]))

model.summary()


def generate_text(model, start_string, temperature, characters_to_generate):
    # Vectorise  start string into numbers
    input_string = [char_to_index[char] for char in start_string]
    input_string = tf.expand_dims(input_string, 0)

    # Empty string to store  generated text
    generated = []

    # (Batch size is 1)
    model.reset_states()
    for i in range(characters_to_generate):
        predictions = model(input_string)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(logits=predictions, num_samples=1)[-1, 0].numpy()

        # Pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_string = tf.expand_dims([predicted_id], 0)

        generated.append(index_to_char[predicted_id])

    return start_string + ''.join(generated)  # generated is a list


# In the arguments, a low temperature gives more predictable text whereas a high temperature gives more random text.
# Also this is where you can change the start string.
generated_text = generate_text(model=model, start_string="The marshes", temperature=0.1, characters_to_generate=1000)
print(generated_text)
