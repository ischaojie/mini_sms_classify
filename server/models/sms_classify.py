# -*- coding: utf-8 -*-
'''
垃圾邮件检测(朴素贝叶斯)

数据说明：
4,827 SMS legitimate messages (86.6%) and a total of 747 (13.4%) spam messages

格式：
ham Go until jurong point, crazy.. Available only in bugis n great world la e buffet... 

ham	Ok lar... Joking wif u oni...

spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. 

ham	U dun say so early hor... U c already then say...
'''

import logging
import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

logging.basicConfig(level=logging.INFO)

base_path = os.path.abspath(os.path.dirname(__file__))
logging.info("base path :{}".format(base_path))


# nltk.download()

def train():
    """
    训练模型
    使用贝叶斯分类
    :return:
    """
    df = pd.read_table(os.path.join(base_path, 'SMSSpamCollection'),
                       sep='\t',
                       header=None,
                       names=['label', 'message'])

    logging.info("dataset ok...")

    # 预处理

    # 文本数据预处理的第一步通常是进行分词，分词后会进行向量化的操作。

    # 将标签转换为数值类型
    df['label'] = df.label.map({'ham': 0, 'spam': 1})
    # 所有邮件小写
    df['message'] = df.message.map(lambda x: x.lower())
    # 移除标点符号
    df['message'] = df.message.str.replace('[^\w\s]', '')

    # 分词
    df['message'] = df['message'].apply(nltk.word_tokenize)

    # 词干抽取
    # 一个单词有很多种时态，将其规范化成一个
    stemmer = PorterStemmer()
    df['message'] = df['message'].apply(lambda x: [stemmer.stem(y) for y in x])

    # Bow模型（词袋模型）
    # 词袋模型假设我们不考虑文本中词与词之间的上下文关系，仅仅只考虑所有词的权重（词在文本中出现的频率）。
    # 分词之后，统计每个词在文本中出现的次数（词频），我们就可以得到该文本基于词的特征，
    # 如果将各个文本样本的这些词与对应的词频放在一起，就是我们常说的向量化。

    # 词向量化

    df['message'] = df['message'].apply(lambda x: ' '.join(x))
    count_vect = CountVectorizer(decode_error="replace")
    counts = count_vect.fit_transform(df['message'])

    # tf-idf
    # 去除词频高但是不重要的词
    # TF-IDF的主要思想是：如果某个词或短语在一篇文章中出现的频率TF高，
    # 并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。
    tfidftransformer = TfidfTransformer()
    tfidf = tfidftransformer.fit_transform(counts)

    # 保存词向量
    feature_vect_path = os.path.join(base_path, 'sms_feature_vact.pkl')
    with open(feature_vect_path, 'wb') as f:
        pickle.dump(count_vect.vocabulary_, f)
    logging.info('词向量已保存...')

    # 保存tf-idf词典
    tfidftransformer_path = os.path.join(base_path, 'sms_tfidftransformer.pkl')
    with open(tfidftransformer_path, 'wb') as f:
        pickle.dump(tfidftransformer, f)
    logging.info('tf-idf已保存...')

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf,
        df['label'],
        test_size=0.1,
        random_state=69
    )

    # 训练
    logging.info('start train model...')
    model = MultinomialNB().fit(X_train, y_train)

    predicted = model.predict(X_test)
    logging.info("accuracy: {}".format(np.mean(predicted == y_test)))
    # 0.9480286738351255

    # 保存训练模型    
    model_file = os.path.join(base_path, "sms_classify_v1.0.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    logging.info("model saved at: {}".format(model_file))


def preprocess(email: str):
    """
    预处理数据
    :param email: 邮件文本
    :return: tf-idf array
    """
    # 预处理
    email = email.lower()
    email = re.sub('[^\w\s]', ' ', email)

    # 分词
    email = nltk.word_tokenize(email)
    logging.info('分词后结果: {}'.format(email[:2]))

    # 词干抽取
    email = [PorterStemmer().stem(e) for e in email]

    # 词频向量化

    email = ' '.join(email)

    # 加载词向量
    feature_vect_path = os.path.join(base_path, 'sms_feature_vact.pkl')
    loaded_vec = CountVectorizer(
        decode_error="replace",
        vocabulary=pickle.load(open(feature_vect_path, "rb"))
    )
    logging.info('词向量加载完成...')

    # 加载tf-idf
    tfidftransformer_path = os.path.join(base_path, 'sms_tfidftransformer.pkl')
    tfidftransformer = pickle.load(open(tfidftransformer_path, "rb"))
    counts_vec = tfidftransformer.transform(loaded_vec.transform([email]))
    logging.info('tf-idf加载完成...')

    return counts_vec.toarray()


def predict(data):
    """
    预测数据
    :param data: 邮件文本
    :return: 预测结果 ham & spam
    """
    labels = ['ham', 'spam']

    data = preprocess(data)

    model_file = os.path.join(base_path, "sms_classify_v1.0.pkl")
    model = pickle.load(open(model_file, 'rb'))

    result = model.predict(data)
    logging.info('predict result: {}'.format(result[0]))

    return labels[result[0]]


if __name__ == "__main__":
    # train()
    email = "Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."
    result = predict(email)
    print(result)
