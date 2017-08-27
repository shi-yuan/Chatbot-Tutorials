import os  # 文件管理
import sys

import numpy as np
import tensorflow as tf
from rankbot import Rankbot
from ranker import Ranker
from textdata import (RankTextData)
from tqdm import tqdm

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/corpus')

args_in = '--device gpu0 ' \
          '--modelTag udc_2l_lr002_dr09_hid256_emb256_len50_vocab10000 ' \
          '--hiddenSize 256 --embeddingSize 256 ' \
          '--vocabularySize 10000 --maxLength 50 ' \
          '--learningRate 0.002 --dropout 0.9 ' \
          '--rootDir C:\\Users\\reade\\Documents\\lecture3 ' \
          '--datasetTag round3_7 --corpus ubuntu'.split()

"""Changed from Rankbot, used for inference
"""


def ranking(args=None, batch_valid=None):
    args = Rankbot.parseArgs(args)
    if not args.rootDir:
        args.rootDir = os.getcwd()

    # 搭建/restore 模型
    graph = tf.Graph()

    with graph.as_default():
        with tf.name_scope('training'):
            model_train = Ranker(args, is_training=True)

        tf.get_variable_scope().reuse_variables()
        with tf.name_scope('validation'):
            model_valid = Ranker(args, is_training=False)

        with tf.name_scope('evluation'):
            model_test = Ranker(args, is_training=False)
            ckpt_model_saver = tf.train.Saver(name='checkpoint_model_saver')
            best_model_saver = tf.train.Saver(name='best_model_saver')

        # 运行session
        # allow_soft_placement = True: 当设置为使用GPU而实际上没有GPU的时候，允许使用其他设备运行。
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        )
        # 恢复指定模型的参数
        ckpt_file = 'C:\\Users\\reade\\Documents\\lecture3\\save\\model-udc_2l_lr002_dr09_hid256_emb256_len50_vocab10000\\best_model.ckpt'
        best_model_saver.restore(sess, ckpt_file)
        valid_acc = [0, 0, 0]
        for nextEvalBatch in tqdm(batches_valid):
            ops, feedDict = model_valid.step(nextEvalBatch)
            loss, eval_summaries = sess.run(ops, feedDict)
            break
            for i in range(3):
                valid_acc[i] += loss[i] / len(batches_valid)
    return loss, nextEvalBatch, valid_acc


args = Rankbot.parseArgs(args_in)
evalData = RankTextData(args)
batches_valid = evalData.getValidBatches()

loss, batch, acc = ranking(args_in, batches_valid)

print(acc)

rank1, rank3, rank5, logits = loss
logits.shape

queries = []
for i in range(len(batch.query_seqs)):
    queries.append(' '.join([evalData.id2word[x] for x in batch.query_seqs[i] if x != 0]))

batch.response_seqs
responses = []
for i in range(len(batch.response_seqs)):
    responses.append(' '.join([evalData.id2word[x] for x in batch.response_seqs[i] if x != 0]))

max_idx = np.argmax(logits, axis=1)
true_pred = []
false_pred = []

for i in range(len(max_idx)):
    if max_idx[i] == 0:
        true_pred.append(i)
    else:
        false_pred.append(i)

eg1 = true_pred[0]
print('query sentence:')
print('\t%s' % queries[eg1])
print('true response:')
print('\t%s' % responses[eg1 * 20])
print('\n')
print('all responses sorted according to scores')

order = np.argsort(logits[eg1])
for i in range(20):
    print('response ranked %d' % i)
    print('\t%s' % responses[eg1 * 20 + order[-i - 1]])

eg2 = true_pred[1]
print('query sentence:')
print('\t%s' % queries[eg2])
print('true response:')
print('\t%s' % responses[eg2 * 20])
print('\n')
print('all responses sorted according to scores')

order = np.argsort(logits[eg2])
for i in range(20):
    print('response ranked %d' % i)
    print('\t%s' % responses[eg2 * 20 + order[-i - 1]])

eg3 = true_pred[2]
print('query sentence:')
print('\t%s' % queries[eg3])
print('true response:')
print('\t%s' % responses[eg3 * 20])
print('\n')
print('all responses sorted according to scores')

order = np.argsort(logits[eg3])
for i in range(20):
    print('response ranked %d' % i)
    print('\t%s' % responses[eg3 * 20 + order[-i - 1]])

eg4 = false_pred[0]
print('query sentence:')
print('\t%s' % queries[eg4])
print('true response:')
print('\t%s' % responses[eg4 * 20])
print('\n')
print('all responses sorted according to scores')

order = np.argsort(logits[eg4])
for i in range(20):
    print('response ranked %d' % i)
    print('\t%s' % responses[eg4 * 20 + order[-i - 1]])

eg5 = false_pred[1]
print('query sentence:')
print('\t%s' % queries[eg5])
print('true response:')
print('\t%s' % responses[eg5 * 20])
print('\n')
print('all responses sorted according to scores')

order = np.argsort(logits[eg5])
for i in range(20):
    print('response ranked %d' % i)
    print('\t%s' % responses[eg5 * 20 + order[-i - 1]])

eg6 = false_pred[2]
print('query sentence:')
print('\t%s' % queries[eg6])
print('true response:')
print('\t%s' % responses[eg6 * 20])
print('\n')
print('all responses sorted according to scores')

order = np.argsort(logits[eg6])
for i in range(20):
    print('response ranked %d' % i)
    print('\t%s' % responses[eg6 * 20 + order[-i - 1]])

eg7 = false_pred[3]
print('query sentence:')
print('\t%s' % queries[eg7])
print('true response:')
print('\t%s' % responses[eg7 * 20])
print('\n')
print('all responses sorted according to scores')

order = np.argsort(logits[eg7])
for i in range(20):
    print('response ranked %d' % i)
    print('\t%s' % responses[eg7 * 20 + order[-i - 1]])
