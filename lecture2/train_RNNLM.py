import numpy as np
import tensorflow as tf
from model import CharRNNLM
from utils import VocabularyLoader, batche2string
import argparse
import os
import logging
import sys
import json
import codecs

TF_VERSION = int(tf.__version__.split('.')[1])


class BatchGenerator(object):

    def __init__(self, tensor_in, tensor_out, batch_size, seq_length):
        """初始化batch产生器
        Input:
            batch_size: 每一个mini-batch里面有多少样本。
            seq_length: 每一个样本的长度，和batch_size一起决定了每个minibatch的数据量。
        """
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.tensor_in = tensor_in
        self.tensor_out = tensor_out

        self.create_batches()
        self.reset_batch_pointer()

    def reset_batch_pointer(self):
        self.pointer = 0

    def create_batches(self):
        self.num_batches = int(self.tensor_in.size / (self.batch_size * self.seq_length))
        self.tensor_in = self.tensor_in[:self.num_batches * self.batch_size * self.seq_length]
        self.tensor_out = self.tensor_out[:self.num_batches * self.batch_size * self.seq_length]

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.x_batches = np.split(self.tensor_in.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(self.tensor_out.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y


class CopyBatchGenerator(BatchGenerator):

    def __init__(self, data, batch_size, seq_length):
        """初始化batch产生器
        Input:
            batch_size: 每一个mini-batch里面有多少样本。
            seq_length: 每一个样本的长度，和batch_size一起决定了每个minibatch的数据量。
        """
        self.batch_size = batch_size
        self.seq_length = seq_length

        tensor_in = np.array(data)
        tensor_out = np.copy(tensor_in)
        tensor_out[:-1] = tensor_in[1:]
        tensor_out[-1] = tensor_in[0]

        super(CopyBatchGenerator, self).__init__(tensor_in, tensor_out, batch_size, seq_length)


def config_train(args=''):
    parser = argparse.ArgumentParser()

    # hyper-parameters to configure the datasets.
    # 数据相关的超参数
    data_args = parser.add_argument_group('Dataset Options')
    data_args.add_argument('--data_file', type = str,
                           default = 'data/ice_and_fire_zh/ice_and_fire_utf8.txt',
                           help = 'data file')
    data_args.add_argument('--encoding', type = str, default = 'utf-8',
                           help = 'the encoding format of data file.')
    data_args.add_argument('--num_unrollings', type=int, default=20,
                           help='number of unrolling steps.')
    data_args.add_argument('--train_frac', type = float, default=0.9,
                           help='fraction of data used for training.')
    data_args.add_argument('--valid_frac', type=float, default=0.05,
                           help='fraction of data used for validation.')
    # test_frac is computed as (1 - train_frac - valid_frac).

    # hyper-parameters to configure the neural network.
    # 模型结构相关的超参数
    network_args = parser.add_argument_group('Model Arch Options')
    network_args.add_argument('--embedding_size', type=int, default=128,
                              help='size of character embeddings, 0 for one-hot')
    network_args.add_argument('--hidden_size', type=int, default=256,
                              help='size of RNN hidden state vector')
    network_args.add_argument('--cell_type', type=str, default='lstm',
                              help='which RNN cell to use (rnn, lstm or gru).')
    network_args.add_argument('--num_layers', type=int, default=2,
                              help='number of layers in the RNN')

    # hyper-parameters to control the training.
    # 训练和优化相关的超参数
    training_args = parser.add_argument_group('Model Training Options')
    # 1. Parameters for iterating through samples
    training_args.add_argument('--num_epochs', type = int, default=50,
                               help='number of epochs')
    training_args.add_argument('--batch_size', type = int, default=20,
                               help='minibatch size')
    # 2. Parameters for dropout setting.
    training_args.add_argument('--dropout', type=float, default=0.0,
                               help='dropout rate, default to 0 (no dropout).')
    training_args.add_argument('--input_dropout', type=float, default=0.0,
                               help=('dropout rate on input layer, default to 0 (no dropout),'
                                     'and no dropout if using one-hot representation.'))
    # 3. Parameters for gradient descent.
    training_args.add_argument('--max_grad_norm', type=float, default=5.,
                               help='clip global grad norm')
    training_args.add_argument('--learning_rate', type=float, default=5e-3,
                               help='initial learning rate')

    # Parameters for manipulating logging and saving models.
    # 学习日志和结果相关的超参数
    logging_args = parser.add_argument_group('Logging Options')
    # 1. Directory to output models and other records.
    logging_args.add_argument('--output_dir', type = str,
                              default = 'demo_model',
                              help = ('directory to store final and'
                                      ' intermediate results and models'))
    # 2. Parameters for printing messages.
    logging_args.add_argument('--progress_freq', type=int, default=100,
                              help=('frequency for progress report in training and evalution.'))
    logging_args.add_argument('--verbose', type=int, default=0,
                              help=('whether to show progress report in training and evalution.'))
    logging_args.add_argument('--debug', dest='debug',action='store_true',
                              help='show debug information')
    logging_args.add_argument('--test', dest='test', action='store_true',
                              help=('parameter for unittesting. Use the first 1000 '
                                    'character to as data to test the implementation'))
    # 3. Parameters to feed in the initial model and current best model.
    logging_args.add_argument('--init_model', type=str,
                              default='', help=('initial model'))
    logging_args.add_argument('--best_model', type=str,
                              default='', help=('current best model'))
    logging_args.add_argument('--best_valid_ppl', type=float,
                              default=np.Inf, help=('current valid perplexity'))
    # 4. Parameters for using saved best models.
    logging_args.add_argument('--init_dir', type=str, default='',
                              help='continue from the outputs in the given directory')

    args = parser.parse_args(args.split())

    return args


def config_sample(args=''):
    parser = argparse.ArgumentParser()

    # hyper-parameters for using saved best models.
    # 学习日志和结果相关的超参数
    logging_args = parser.add_argument_group('Logging_Options')
    logging_args.add_argument('--init_dir', type=str,
                              default='demo_model/',
                              help='continue from the outputs in the given directory')

    # hyper-parameters for sampling.
    # 设置sampling相关的超参数
    testing_args = parser.add_argument_group('Sampling Options')
    testing_args.add_argument('--max_prob', dest='max_prob', action='store_true',
                              help='always pick the most probable next character in sampling')
    testing_args.set_defaults(max_prob=False)

    testing_args.add_argument('--start_text', type=str,
                              default='The meaning of life is ',
                              help='the text to start with')

    testing_args.add_argument('--length', type=int,
                              default=100,
                              help='length of sampled sequence')

    testing_args.add_argument('--seed', type=int,
                              default=-1,
                              help=('seed for sampling to replicate results, '
                                    'an integer between 0 and 4294967295.'))

    args = parser.parse_args(args.split())

    return args

args = config_train('--debug --verbose 1')
vars(args)

# 在目标文件夹（output_dir)里面创建save_model, best_model 和 tensorboard_log
# 分别保存
#   1. 训练过程中的中间模型
#   2. 目前最好的模型
#   3. 用于tensorboard可视化的日志文件

args.save_model = os.path.join(args.output_dir, 'save_model/model')
args.save_best_model = os.path.join(args.output_dir, 'best_model/model')
args.tb_log_dir = os.path.join(args.output_dir, 'tensorboard_log/')
args.vocab_file = ''

# 小心使用，如果目标文件夹已经存在，先将其删除
print('=' * 80)
print('All final and intermediate outputs will be stored in %s/' % args.output_dir)
print('=' * 80 + '\n')
if os.path.exists(args.output_dir):
    import shutil
    shutil.rmtree(args.output_dir)

# 建立目标文件夹和子文件夹
for paths in [args.save_model, args.save_best_model, args.tb_log_dir]:
    os.makedirs(os.path.dirname(paths))


print('模型将保存在：%s' % args.save_model)
print('最好的模型将保存在：%s' % args.save_best_model)
print('tensorboard相关文件将保存在：%s' % args.tb_log_dir)



logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.INFO, datefmt='%I:%M:%S')
logging

if args.debug:
    logging.info('args are:\n%s', args)

if len(args.init_dir) != 0:
    with open(os.path.join(args.init_dir, 'result.json'), 'r') as f:
        result = json.load(f)

    params = result['params']
    args.init_model = result['latest_model']
    best_model = result['best_model']
    best_valid_ppl = result['best_valid_ppl']

    if 'encoding' in result:
        args.encoding = result['encoding']
    else:
        args.encoding = 'utf-8'
    args.vocab_file = os.path.join(args.init_dir, 'vocab.json')
else:
    params = {'batch_size': args.batch_size,
              'num_unrollings': args.num_unrollings,
              'hidden_size': args.hidden_size,
              'max_grad_norm': args.max_grad_norm,
              'embedding_size': args.embedding_size,
              'num_layers': args.num_layers,
              'learning_rate': args.learning_rate,
              'cell_type': args.cell_type,
              'dropout': args.dropout,
              'input_dropout': args.input_dropout}
    best_model = ''

params = {'batch_size': args.batch_size,
          'num_unrollings': args.num_unrollings,
          'hidden_size': args.hidden_size,
          'max_grad_norm': args.max_grad_norm,
          'embedding_size': args.embedding_size,
          'num_layers': args.num_layers,
          'learning_rate': args.learning_rate,
          'cell_type': args.cell_type,
          'dropout': args.dropout,
          'input_dropout': args.input_dropout}
s_params= json.dumps(params, sort_keys=True, indent = 4)
logging.info('\n\nParameters are:\n%s\n', s_params)

print(args.data_file)


# codecs: python的编码和解码模块, 可以参考
# http://blog.csdn.net/iamaiearner/article/details/9138865
# http://www.cnblogs.com/TsengYuen/archive/2012/05/22/2513290.html
# http://blog.csdn.net/suofiya2008/article/details/5579413
# 等博客
# Read and split data.
logging.info('Reading data from: %s', args.data_file)
with codecs.open(args.data_file, encoding=args.encoding) as f:
    text = f.read()

if args.test:
    text = text[:50000]
logging.info('Number of characters: %s', len(text))

if args.debug:
    logging.info('First %d characters: %s', 10, text[:10])

# 将整本数据集分割成train, test, validation数据集
logging.info('Creating train, valid, test split')
train_size = int(args.train_frac * len(text))
valid_size = int(args.valid_frac * len(text))
test_size = len(text) - train_size - valid_size
train_text = text[:train_size]
valid_text = text[train_size:train_size + valid_size]
test_text = text[train_size + valid_size:]

# 建立词典
vocab_loader = VocabularyLoader()
if len(args.vocab_file) != 0:
    vocab_loader.load_vocab(args.vocab_file, args.encoding)
else:
    logging.info('Creating vocabulary')
    vocab_loader.create_vocab(text)
    vocab_file = os.path.join(args.output_dir, 'vocab.json')
    vocab_loader.save_vocab(vocab_file, args.encoding)
    logging.info('Vocabulary is saved in %s', vocab_file)
    args.vocab_file = vocab_file

params['vocab_size'] = vocab_loader.vocab_size
logging.info('Vocab size: %d\n', vocab_loader.vocab_size)

logging.info('sampled words in vocabulary %s ' % list(zip(list(vocab_loader.index_vocab_dict.keys())[:1000:100],
                                                          list(vocab_loader.index_vocab_dict.values())[:1000:100])))

# Create batch generators.
batch_size = params['batch_size']
num_unrollings = params['num_unrollings']

train_batches = CopyBatchGenerator(list(map(vocab_loader.vocab_index_dict.get, train_text)), batch_size, num_unrollings)
valid_batches = CopyBatchGenerator(list(map(vocab_loader.vocab_index_dict.get, valid_text)), batch_size, num_unrollings)
test_batches = CopyBatchGenerator(list(map(vocab_loader.vocab_index_dict.get, test_text)), batch_size, num_unrollings)

logging.info('Test the batch generators')
x, y = train_batches.next_batch()
print(x[0])
logging.info((str(x[0]), str(batche2string(x[0], vocab_loader.index_vocab_dict))))
logging.info((str(y[0]), str(batche2string(y[0], vocab_loader.index_vocab_dict))))


# 建立训练，valid，测试 对象
logging.info('Creating graph')
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('training'):
        train_model = CharRNNLM(is_training=True, infer=False, **params)

    tf.get_variable_scope().reuse_variables()
    with tf.name_scope('validation'):
        valid_model = CharRNNLM(is_training=False, infer=False, **params)

    with tf.name_scope('evaluation'):
        test_model = CharRNNLM(is_training=False, infer=False, **params)
        saver = tf.train.Saver(name='model_saver')
        best_model_saver = tf.train.Saver(name='best_model_saver')

logging.info('Start training\n')

result = {}
result['params'] = params
result['vocab_file'] = args.vocab_file
result['encoding'] = args.encoding


try:
    with tf.Session(graph=graph) as session:
        graph_info = session.graph_def

        train_writer = tf.summary.FileWriter(args.tb_log_dir + 'train/', graph_info)
        valid_writer = tf.summary.FileWriter(args.tb_log_dir + 'valid/', graph_info)

        # load a saved model or start from random initialization.
        if len(args.init_model) != 0:
            saver.restore(session, args.init_model)
        else:
            tf.global_variables_initializer().run()

        learning_rate = args.learning_rate
        for epoch in range(args.num_epochs):
            logging.info('=' * 19 + ' Epoch %d ' + '=' * 19 + '\n', epoch)
            logging.info('Training on training set')

            # 第一部分.
            # training step, running one epoch on training data
            ppl, train_summary_str, global_step = train_model.run_epoch(
                session, train_batches, is_training=True,
                learning_rate=learning_rate, verbose=args.verbose,
                freq=args.progress_freq)
            # record the summary
            train_writer.add_summary(train_summary_str, global_step)
            train_writer.flush()
            # save model
            # 注意：save操作在session内部才有意义！
            saved_path = saver.save(session, args.save_model, global_step=train_model.global_step)
            logging.info('Latest model saved in %s\n', saved_path)

            # 第二部分.
            # evaluation step, running one epoch on validation data
            logging.info('Evaluate on validation set')
            valid_ppl, valid_summary_str, _ = valid_model.run_epoch(
                session, valid_batches, is_training=False,
                learning_rate=learning_rate, verbose=args.verbose,
                freq=args.progress_freq)
            # save and update best model
            if (len(best_model) == 0) or (valid_ppl < best_valid_ppl):
                best_model = best_model_saver.save(
                    session, args.save_best_model,
                    global_step=train_model.global_step)
                best_valid_ppl = valid_ppl
            else:
                learning_rate /= 2.0
                logging.info('Decay the learning rate: ' + str(learning_rate))
            valid_writer.add_summary(valid_summary_str, global_step)
            valid_writer.flush()
            logging.info('Best model is saved in %s', best_model)
            logging.info('Best validation ppl is %f\n', best_valid_ppl)

            # 第三部分.
            # update readable summary
            result['latest_model'] = saved_path
            result['best_model'] = best_model
            # Convert to float because numpy.float is not json serializable.
            result['best_valid_ppl'] = float(best_valid_ppl)

            result_path = os.path.join(args.output_dir, 'result.json')
            if os.path.exists(result_path):
                os.remove(result_path)
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2, sort_keys=True)

        logging.info('Latest model is saved in %s', saved_path)
        logging.info('Best model is saved in %s', best_model)
        logging.info('Best validation ppl is %f\n', best_valid_ppl)

        logging.info('Evaluate the best model on test set')
        saver.restore(session, best_model)
        test_ppl, _, _ = test_model.run_epoch(session, test_batches, is_training=False,
                                              learning_rate=learning_rate, verbose=args.verbose, freq=args.progress_freq)
        result['test_ppl'] = float(test_ppl)
finally:
    result_path = os.path.join(args.output_dir, 'result.json')
    if os.path.exists(result_path):
        os.remove(result_path)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)

args = config_sample('--init_dir demo_model --length 100 --start_text 之后')
vars(args)

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.INFO, datefmt='%I:%M:%S')

# Prepare parameters.
with open(os.path.join(args.init_dir, 'result.json'), 'r') as f:
    result = json.load(f)
params = result['params']
best_model = result['best_model']
best_valid_ppl = result['best_valid_ppl']
if 'encoding' in result:
    args.encoding = result['encoding']
else:
    args.encoding = 'utf-8'

args.vocab_file = os.path.join(args.init_dir, 'vocab.json')
vocab_loader = VocabularyLoader()
vocab_loader.load_vocab(args.vocab_file, args.encoding)

logging.info('best_model: %s\n', best_model)

# Create graphs
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('evaluation'):
        model = CharRNNLM(is_training=False, infer=True, **params)
    saver = tf.train.Saver(name='model_saver')

if args.seed >= 0:
    np.random.seed(args.seed)
with tf.Session(graph=graph) as session:
    saver.restore(session, best_model)
    sample = model.sample_seq(session, args.length, args.start_text, vocab_loader,
                              max_prob=args.max_prob)
    print('Sampled text is:\n\n%s' % sample)

