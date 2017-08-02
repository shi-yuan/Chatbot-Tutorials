import tensorflow as tf

batch_size = 16
dropout = 0.0
embedding_size = 64
hidden_size = 128
input_dropout = 0.0
learning_rate = 0.005
max_grad_norm = 5.0
model = 'rnn'
num_layers = 2
num_unrollings = 10
vocab_size = 26
is_training = True

sess = tf.InteractiveSession()

if model == 'rnn':
    cell_fn = tf.nn.rnn_cell.BasicRNNCell
elif model == 'lstm':
    cell_fn = tf.nn.rnn_cell.BasicLSTMCell
elif model == 'gru':
    cell_fn = tf.nn.rnn_cell.GRUCell

params = dict()
if model == 'lstm':
    params['forget_bias'] = 1.0  # 1.0 is default value
cell = cell_fn(hidden_size, **params)

cells = [cell]
for i in range(num_layers - 1):
    higher_layer_cell = cell_fn(hidden_size, **params)
    cells.append(higher_layer_cell)

multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

multi_cell.zero_state(batch_size, tf.float32)

with tf.name_scope('initial_state'):
    zero_state = multi_cell.zero_state(batch_size, tf.float32)
    if model == 'rnn' or model == 'gru':
        initial_state = tuple(
            [tf.placeholder(tf.float32,
                            [batch_size, multi_cell.state_size[idx]],
                            'initial_state_' + str(idx + 1))
             for idx in range(num_layers)])
    elif model == 'lstm':
        initial_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(
                tf.placeholder(tf.float32, [batch_size, multi_cell.state_size[idx][0]],
                               'initial_lstm_state_' + str(idx + 1)),
                tf.placeholder(tf.float32, [batch_size, multi_cell.state_size[idx][1]],
                               'initial_lstm_state_' + str(idx + 1)))
                for idx in range(num_layers)])
