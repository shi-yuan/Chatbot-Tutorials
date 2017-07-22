import collections
import math
import re
from collections import Counter

import gensim
import jieba
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# 调用 jieba分词module后，添加单词本（人名等）:
jieba.load_userdict("data/names_separate.txt")

# 读取数据
filename = 'data/ice_and_fire_utf8.txt'
text_lines = []
with open(filename, 'r', encoding="utf-8") as f:
    for line in tqdm(f):
        text_lines.append(line)
print('总共读入%d行文字' % (len(text_lines)))

# 分词
# data_words: 训练我们的cbow和skip-gram模型
data_words = []
# data_lines: 调用gensim.word2vec训练word2vec模型
data_lines = []
for line in tqdm(text_lines):
    one_line = [' '.join(jieba.cut(line, cut_all=False))][0].split(' ')
    data_words.extend(one_line)
    data_lines.append(one_line)

# 去掉标点和数字
# 标点符号 (punctuation)
punct = set(
    u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．＊：；Ｏ？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…０１２３４５６７８９''')
isNumber = re.compile(r'\d+.*')

filter_words = [w for w in data_words if (w not in punct) and (not isNumber.search(w.lower()))]

# 建立词典
vocabulary_size = 30000


def build_vocab(words):
    """对文字数据中最常见的单词建立词典

    Arguments:
        words: 一个list的单词,来自分词处理过的文本数据库.

    Returns:
        data: 输入words数字编码后的版本
        count: dict, 单词 --> words中出现的次数
        dictionary: dict, 单词 --> 数字ID的字典
        reverse_dictionary: dict, 数字ID-->单词的字典
    """
    # 1. 统计每个单词的出现次数
    words_counter = Counter(words)
    # 2. 选取常用词
    count = [['UNK', -1]]
    count.extend(words_counter.most_common(vocabulary_size - 1))
    # 3. 词语编号
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    # 4. 引入特殊词语UNK
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)

    print(unk_count)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


# 生成词典
data, count, dictionary, reverse_dictionary = build_vocab(filter_words)

# 打印一些信息，简单地人工检查有没有明显错误
print('共有 %d 万个词语' % (len(filter_words) // 10000))
print('前十个单词是 %s' % ' '.join(filter_words[:10]))

# 根据出现频率排列单词
words_counter = Counter(filter_words)
word_sorted = []
for (k, v) in words_counter.most_common(len(words_counter)):
    word_sorted.append((k, v))

# 看有哪些词，词频多少
# 是否常见词的词频高，生僻词的词频低
print('最常见的5个词，包括换行符:')
for i in range(5):
    print('%d: %s %d' % (i, word_sorted[i][0], word_sorted[i][1]))

print('\n')
print('出现频率居中的一些词语:')
for i in range(1111, 2000, 100):
    print('%d: %s %d' % (i, word_sorted[i][0], word_sorted[i][1]))

print('\n')
print('出现频率万名开外的一些词语')
for i in range(10000, len(word_sorted), 10000):
    print('%d: %s %d' % (i, word_sorted[i][0], word_sorted[i][1]))

# 验证输入jieba的一些name entity被正确地切分出来
demo_names = ['史塔克', '兰尼斯特', '龙之母']
for name in demo_names:
    print('%s 出现 %d 次' % (name, words_counter[name]))

# 验证单词到数字的编码是正确的
demo_num = data[1000:1100]
print(demo_num)

demo_str = ''
for i in range(1000, 1100):
    demo_str = demo_str + (reverse_dictionary[data[i]]) + ' '
print(demo_str)

# 使用word2vec训练模型
model = gensim.models.Word2Vec(iter=1)
model.build_vocab(data_lines)
model.train(data_lines, total_examples=len(data_lines), epochs=10)

test_words = ['史塔克', '提利昂', '琼恩', '长城', '衣物', '力量', '没关系']
neighbors = []
for test_word in test_words:
    neighbors.append(model.most_similar(test_word))

for i in range(len(neighbors)):
    str = ' '.join([x[0] for x in neighbors[i]])
    print('%s:' % test_words[i])
    print('\t%s\n' % (str))

####################
# skip-gram模型
####################
data_index = 0
np.random.seed(0)


def generate_batch_sg(data, batch_size, num_skips, slide_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * slide_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    # 滑动窗：[ slide_window target slide_window ]，宽度为 span
    span = 2 * slide_window + 1
    buffer = collections.deque(maxlen=span)

    # 扫过文本，将一个长度为 2*slide_window+1 的滑动窗内的词语放入buffer
    # buffer里面，居中的是target，“当前单词”
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # 下面的 for 循环：
    # 在产生一个有batch_size个样本的minibatch的时候，
    # 我们选择 batch_size//num_skips 个 “当前单词” （或者叫“目标单词”）
    # 并且从“当前单词”左右的2*slide_window 个词语组成的context里面选择num_skips个单词
    # “当前单词”（我们叫做x）和num_skips个单词中的每一个（我们叫做y_i）组成一个监督学习样本：
    # 给定单词x, 在它的context里面应该大概率地出现单词y_i
    for i in range(batch_size // num_skips):
        # 在每个长度为2*slide_window + 1 的滑动窗里面，
        # 我们选择num_skips个（“当前单词”，“语境单词”）的组合
        # 为了凑齐batch_size个组合，我们需要batch_size//num_skips个滑动窗
        rand_x = np.random.permutation(span)
        j, k = 0, 0
        for j in range(num_skips):
            while rand_x[k] == slide_window:
                k += 1
            batch[i * num_skips + j] = buffer[slide_window]
            labels[i * num_skips + j, 0] = buffer[rand_x[k]]
            k += 1

        # 将滑动窗向右滑动随机步
        rand_step = np.random.randint(1, 5)
        for _ in range(rand_step):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)

    return batch, labels


# 测试代码：
batch_size = 8
for num_skips, slide_window in [(2, 1), (4, 2)]:
    batch, labels = generate_batch_sg(data=data,
                                      batch_size=batch_size,
                                      num_skips=num_skips,
                                      slide_window=slide_window)
    print('\nwith num_skips = %d and slide_window = %d:' % (num_skips, slide_window))
    for i in range(batch_size):
        print('%s --> %s' % (reverse_dictionary[batch[i]],
                             reverse_dictionary[labels[i][0]]))

batch_size = 128
num_sampled = 64  # 近似计算cross entropy loss的时候的negative examples参数
embedding_size = 128  # Dimension of the embedding vector

# skip-gram的两个重要的hyperparameters:
# 1. 语境范围：考虑和周围多少个词语的共存关系
slide_window = 1
# 2.　”使用频率“：（当前单词，临近单词）的组合使用多少次
num_skips = 1  # How many times to reuse an input to generate a label

# 产生测试数据
valid_size = 8
valid_examples = list(np.random.permutation(1000)[:valid_size])
names = ['史塔克', '提利昂', '琼恩', '长城', '南方', '死亡', '家族', '支持', '愤怒']
for name in names:
    valid_examples.append(dictionary[name])
    valid_size += 1

graph_sg = tf.Graph()

with graph_sg.as_default():  # , tf.device('/cpu:0'):

    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                   labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable quantities
    # that contribute to the tensor it is passed.
    # See docs on `tf.train.Optimizer.minimize()` for more details.
    # optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)
    optimizer = tf.train.AdagradOptimizer(learning_rate=3.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

num_steps = 100001

with tf.Session(graph=graph_sg) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch_sg(data, batch_size, num_skips, slide_window)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0

        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    final_embeddings_sg = normalized_embeddings.eval()
