import jieba, os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.externals import joblib


def tf_idf():
    """
    评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。
    去除出现的多的无用词
    :return:
    """
    # 读取文件，将文件整理成需要的的格式
    name = os.listdir('./data/origin_train/')
    path_name_li = ['./data/origin_train/' + i for i in name]
    sentence_li = []
    for i in path_name_li:
        with open(i, 'r', encoding='utf-8') as f:
            content = f.read().replace('\n', '').replace('\u3000', '')
            sentence_li.append(content)
    print(sentence_li)
    print(len(sentence_li))

    # 返回的是分词结果的迭代器
    split_sentence_li = []
    for sentence in sentence_li:
        sentence = jieba.cut(sentence)
        sentence = ' '.join(list(sentence))
        split_sentence_li.append(sentence)
    print(split_sentence_li)

    # 提取特征词
    tf = TfidfVectorizer()
    ret = tf.fit_transform(split_sentence_li)

    # 将特征词写入文件
    print(tf.get_feature_names())
    all_li = tf.get_feature_names()
    with open('./data/input_data/all_feature_name_li.txt', 'w', encoding='utf-8') as f2:
        f2.write(str(all_li))

    print(ret.toarray())
    # print(tf.inverse_transform(ret))
    # 生成特征向量
    data = ret.toarray()
    # 生成标签向量【前80个非it行业，标签值为0，后80个为it行业，标签值为1】
    label = np.array([i // 80 for i in range(0, 160)])
    # print(label)
    # 切分训练集何测试集
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.3)

    # 训练并保存模型
    # model = GaussianNB()
    # model.fit(train_x, train_y)
    # print(model)
    # joblib.dump(model, "./data/input_data/train_model.model")

    # 读取模型
    model = joblib.load("./data/input_data/train_model.model")
    expected = test_y
    print(len(test_x[0]))
    # 预测结果
    predicted = model.predict(test_x)
    print(predicted)
    print(expected)
    print(metrics.classification_report(expected, predicted))  # 输出分类信息
    labels = list(set(label))  # 去重复，得到标签类别
    print(metrics.confusion_matrix(expected, predicted, labels=labels))


def predict():
    input_file = './data/input_data/wiki_all_2.json'
    all_li_file = './data/input_data/all_feature_name_li.txt'

    output_file = './data/output_data/pure_wiki.sql'
    output2_file = './data/output_data/not_pure_wiki.sql'
    with open(output_file, 'w', encoding='utf-8') as f, open(output2_file, 'w', encoding='utf-8') as f2, \
            open(all_li_file, 'r', encoding='utf-8') as f0, open(input_file, 'r', encoding='utf-8') as f1:

        all_li = eval(f0.read())
        for i in f1:
            line = eval(i)
            setn = line['introduction']
            setn = re.sub('[a-zA-Z（ ）\[\]]*', '', setn)

            setn_li = list(jieba.cut(setn))

            # 根据已有的特征词的列表，来创建每个要预测的文本的向量
            sim_li = []
            for j in all_li:
                if j in setn_li:
                    sim_li.append(1)
                else:
                    sim_li.append(0)
            # print(sim_li)
            # 预测该文本的分类
            sil = np.array([sim_li])
            # print(sil)
            model = joblib.load("./data/input_data/train_model.model")
            predicted = model.predict(sil)
            print(predicted)

            # 写入文件
            if str(predicted) == '[1]':
                f.write(i)
            else:
                f2.write(i)


if __name__ == '__main__':
    tf_idf()
    predict()
