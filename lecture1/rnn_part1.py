import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print(tf.__version__)

# 设置合成数据的特征
total_size = 1000000


def gen_data(size=total_size):
    """ 按照上图生成合成序列数据

    Arguments:
        size: input 和 output 序列的总长度

    Returns:
        X, Y: input 和 output 序列，rank-1的numpy array （即，vector)
    """

    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i - 3] == 1:
            threshold += 0.5
        if X[i - 8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)


# adapted from https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size, num_steps):
    """产生minibatch数据

    Arguments:
        raw_data: 所有的数据， (input, output) tuple
        batch_size: 一个minibatch包含的样本数量；每个样本是一个sequence
        num_step: 每个sequence样本的长度

    Returns:
        一个generator，在一个tuple里面包含一个minibatch的输入，输出序列
    """

    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)


def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)


"""
Placeholders
"""
tf.reset_default_graph()

batch_size = 32
num_steps = 4

x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

"""
RNN Inputs， 将前面定义的placeholder输入到RNN cells
"""

# 将输入序列中的每一个0,1数字转化为二维one-hot向量
num_classes = 2
x_one_hot = tf.one_hot(x, num_classes)  # [batch_size, num_steps，num_classes = 2]
rnn_inputs = tf.unstack(x_one_hot, axis=1)  # [ num_steps, [batch_size, num_classes]]

"""
手动实现 rnn_cell

This is very similar to the __call__ method on Tensorflow's BasicRNNCell. See:
https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/rnn_cell_impl.py#L232
（见上图）
"""

state_size = 16

with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W', [num_classes + state_size, state_size],
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
    b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))


def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [num_classes + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)


"""
对每个time frame应用rnn
"""

rnn_outputs = []
init_state = tf.zeros([batch_size, state_size])

state = init_state
for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)

final_state = rnn_outputs[-1]

"""
计算损失函数，定义优化器
"""
# 从每个 time frame 的 hidden state
# 映射到每个 time frame 的最终 output（prediction）；
# 和CBOW或者SKIP-GRAM的最上一层相同

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

# 计算损失函数
y_as_list = tf.unstack(y, num=num_steps, axis=1)
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for
          logit, label in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses)

# 定义优化器
learning_rate = 0.1
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

# 本模型所有的variable：
all_vars = [node.name for node in tf.global_variables()]
for var in all_vars:
    print(var)

# 打印graph的nodes：
all_node_names = [node for node in tf.get_default_graph().as_graph_def().node]
# 或者：
# all_node_names = [node for node in tf.get_default_graph().get_operations()]
all_node_values = [node.values() for node in tf.get_default_graph().get_operations()]

for i in range(0, len(all_node_values), 50):
    print('output and operation %d:' % i)
    print(all_node_values[i])
    print('-------------------')
    print(all_node_names[i])
    print('\n')
    print('\n')

for i in range(len(all_node_values)):
    print('%d: %s' % (i, all_node_values[i]))

for i in range(len(all_node_values)):
    if 'op: "Add"' in repr(all_node_names[i]):
        print('output and operation %d:' % i)
        print(all_node_values[i])
        print('-------------------')
        print(all_node_names[i])
        print('\n')
        print('\n')

"""
训练模型的参数
"""

num_epochs = 4
verbose = True

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    training_losses = []
    for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
        training_loss = 0
        training_state = np.zeros((batch_size, state_size))
        if verbose:
            print("\nEPOCH", idx)
        for step, (X, Y) in enumerate(epoch):
            tr_losses, training_loss_, training_state, _ = sess.run(
                [losses, total_loss, final_state, train_step],
                feed_dict={x: X, y: Y, init_state: training_state})
            training_loss += training_loss_
            if step % 500 == 0 and step > 0:
                if verbose:
                    print("At step %d, average loss of last 500 steps are %f\n"
                          % (step, training_loss / 500.0))
                training_losses.append(training_loss / 500.0)
                training_loss = 0

plt.plot(training_losses)  # when num_len = 4, state_size = 16
