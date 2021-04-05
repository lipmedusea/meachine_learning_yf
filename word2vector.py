# 导入一些需要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pymysql
import os
import pandas as pd
import jieba

import collections
import math
import os
import random
import zipfile

import numpy as np
import pandas as pd
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from string import punctuation
import sys

add_punc='，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'

all_punc=punctuation+add_punc

from operator import mul
from functools import reduce
reduce(mul, range(1, 5))

def mull(df):
    mull = []
    for x in range(df.shape[1]):
        y = reduce(mul, df[:, x])
        mull.append(y)
    return mull



def load_data_new(sql,filename="df_load_new.csv"):
    save_dir = 'load_data/'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    load_path = save_dir
    filenames = os.listdir(load_path)
    filepath = save_dir + filename
    if filename not in filenames:
        # conn = pymysql.connect(host="bi-private.hfjy.com", port=33333, user="data_hfjy", passwd="data_hfjy_123456",
        #                        db="hfSB", charset="udf8")
        conn = pymysql.connect(host="rm-2ze974348wa9e1ev3uo.mysql.rds.aliyuncs.com", port=3306, user="yanfei_read",
                               passwd="DxIqy0R9rzT95yBfstgx", db="hfjydb", charset="utf8")
        df = pd.read_sql(sql, conn)
        conn.close()
        df.to_csv(filepath, index=None, encoding="GB18030")
    else:
        df = pd.read_csv(filepath, encoding="GB18030")
    return df

sql = """ select s.student_id , cr.end_time, cr.content
from view_student s left join view_communication_record cr on s.student_intention_id = cr.student_intention_id
where role_code like 'ZJ%' 
and s.student_id not in (28475, 82411)
and s.student_id is not null
and cr.content is not null
and cr.content not like '%批处理%'
and cr.content not like '%批量%'
and cr.content not like '%系统%'
and cr.content not like '%测试%'
and cr.content not like '%刘莹%'
and cr.content not like '%没人接%'
and cr.content not like '%未接通%'
and cr.content not like '%关机%'
and cr.content not like '%挂断%'
and cr.content not like '%非常%'
and cr.content not like '%地图%'
and cr.content not like '%没接通%'
and cr.content not like '%用户正忙%'
and cr.content not like '%空号%'
and cr.content not like '%废弃%'
and cr.content not like '%关闭%'
and cr.content not like '%未接%'
and cr.content not like '%拒接%'
and cr.content not like '%不接%'
and cr.content not like '%没接%'
and cr.content not like '%未接听%'
and cr.content not like '%电话%'
and cr.content not like '%无人%'
and cr.content not like '%忙线%'
and cr.content not like '%微信%'
 """

import re
quoto = '\d{1,}|[’!#$%&\()+,、.-:;<=>?@[ \\\\ \]^_`{}~，、。：【】；”“？（）{ }{}\n\s\t/]|[a-z]|[A-Z]'

#分词保存
df = load_data_new(sql=sql, filename="df_text_all.csv")
df["contents"] = [re.sub(quoto, '', x) for x in df["content"]]
df = df[df["contents"]!=""]

# li = []
df["cut"]=[jieba.lcut(x, cut_all=False) for x in df["contents"]]

# df.to_csv("load_data/df_text_jieba_stop.csv")
#
#读取分词数据

# df = pd.read_csv("load_data/df_text_jieba_stop.csv")
# df["cuts"] = [str(x).replace("[", "") for x in df["cut"]]
# df["cuts"] = [x.replace("]","") for x in df["cuts"]]
# df["cuts"] = [x.replace("'","") for x in df["cuts"]]
# df["cuts"] = [x.split(",") for x in df["cut"]]

text_list = []
for x in df["cut"]:
    text_list.extend(x)

# text_list.remove(" ：")
# text_list.remove(" \\n")
# text_list.remove(" \\r\\n")
# text_list.remove(" .")
# text_list.remove("  ")
# text_list.remove(" 】")
# text_list.remove("【")

#制作此表
def build_dataset(words, n_words):
  """
  函数功能：将原始的单词表示变成index
  """
  unk_word= []
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      unk_word.append(word)
      index = 0  # UNK的index为0
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary, unk_word

# 第二步: 制作一个词表，将不常见的词变成一个UNK标识符
# 词表的大小为5万（即我们只考虑最常出现的5万个词）
vocabulary_size = 50000


data, count, dictionary, reverse_dictionary, unk_word = build_dataset(text_list,
                                                            vocabulary_size)
# del text_list  # 删除已节省内存
# 输出最常出现的5个单词
print('Most common words (+UNK)', count[:5])
# 输出转换后的数据库data，和原来的单词（前10个）
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
# 我们下面就使用data来制作训练集
data_index = 0




# 第三步：定义一个函数，用于生成skip-gram模型用的batch
def generate_batch(batch_size, num_skips, skip_window):
  # data_index相当于一个指针，初始为0
  # 每次生成一个batch，data_index就会相应地往后推
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  # data_index是当前数据开始的位置
  # 产生batch后就往后推1位（产生batch）
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    # 利用buffer生成batch
    # buffer是一个长度为 2 * skip_window + 1长度的word list
    # 一个buffer生成num_skips个数的样本
#     print([reverse_dictionary[i] for i in buffer])
    target = skip_window  # target label at the center of the buffer
#     targets_to_avoid保证样本不重复
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    # 每利用buffer生成num_skips个样本，data_index就向后推进一位
    data_index = (data_index + 1) % len(data)
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels



batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])



# 第四步: 建立模型.

batch_size = 128
embedding_size = 128  # 词嵌入空间是128维的。即word2vec中的vec是一个128维的向量
skip_window = 1       # skip_window参数和之前保持一致
num_skips = 2         # num_skips参数和之前保持一致

# 在训练过程中，会对模型进行验证
# 验证的方法就是找出和某个词最近的词。
# 只对前valid_window的词进行验证，因为这些词最常出现
valid_size = 16     # 每次验证16个词
valid_window = 100  # 这16个词是在前100个最常见的词中选出来的
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# 构造损失时选取的噪声词的数量
num_sampled = 64

graph = tf.Graph()

with graph.as_default():

  # 输入的batch
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  # 用于验证的词
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # 下面采用的某些函数还没有gpu实现，所以我们只在cpu上定义模型
  with tf.device('/cpu:0'):
    # 定义1个embeddings变量，相当于一行存储一个词的embedding
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # 利用embedding_lookup可以轻松得到一个batch内的所有的词嵌入
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # 创建两个变量用于NCE Loss（即选取噪声词的二分类损失）
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # tf.nn.nce_loss会自动选取噪声词，并且形成损失。
  # 随机选取num_sampled个噪声词
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # 得到loss后，我们就可以构造优化器了
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # 计算词和词的相似度（用于验证）
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  # 找出和验证词的embedding并计算它们和所有单词的相似度
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # 变量初始化步骤
  init = tf.global_variables_initializer()


# 第五步：开始训练
num_steps = 10001

with tf.Session(graph=graph) as session:
  # 初始化变量
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # 优化一步
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # 2000个batch的平均损失
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # 每1万步，我们进行一次验证
    if step % 10000 == 0:
      # sim是验证词与所有词之间的相似度
      sim = similarity.eval()
      # 一共有valid_size个验证词
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # 输出最相邻的8个词语
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  # final_embeddings是我们最后得到的embedding向量
  # 它的形状是[vocabulary_size, embedding_size]
  # 每一行就代表着对应index词的词嵌入表示
  final_embeddings = normalized_embeddings.eval()



# #生成文本向量
txt = np.unique(text_list)
unk_dict = {}
for x in unk_word:
    print(x)
    unk_dict[x] = 0

dictionary.update(unk_dict)

# vec_dict = {}
# for x in txt:
#     print(x)
#     if x in unk_word:
#         vec_dict[x] = final_embeddings[0]
#     else:
#         vec_dict[x] = final_embeddings[dictionary[x]]
#
# print(txt[0])
def squ(x):
    y = dictionary[x]
    print(y)
    return final_embeddings[y]

import datetime
start_time = datetime.datetime.now()
t = np.array([np.array(list(map(squ, x))).sum(axis=0) for x in df["cut"]])

# t_mul = [np.array(list(map(squ, x))) for x in df["cut"]]
# t_mul[0]
# t = np.array([mull(x) for x in t_mul])

end_time = datetime.datetime.now()
print(end_time-start_time)

# for x in range(t.shape[1]):
#     print(x)
#     feature_txt = "feature_" + str(x)
#     df[feature_txt] = list(t[:,x])
#

df = pd.concat([df, pd.DataFrame(t)], axis=1)


#teacher_student对应关系
sqll = """select 
distinct lp.teacher_id,lp.student_id
from lesson_plan lp 
where lp.teacher_id in 
(
select distinct teacher_id from bidata.trail_boost_wdf
) and lesson_type = 1 and status in (3,5) and solve_status <> 6
"""
teacher_to_student = load_data_new(sql=sqll, filename="teacher_student")


dfs = pd.merge(teacher_to_student, df , on="student_id", how ="left")
dfs = dfs[~dfs["teacher_id"].isnull()]
dfs = dfs[~dfs["content"].isnull()]
#按照teacher_id聚合
columns = list(range(t.shape[1]))

d_teacher = dfs[columns].groupby(dfs['teacher_id']).sum()
d_teacher["teacher_id"] = d_teacher.index
# d_teacher.to_csv("load_data/teacher_vector_sum.csv")
