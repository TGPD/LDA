#author:陈楚翘
#stuID:1811470

import traceback
import re
import gensim
from gensim.models import word2vec
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
#建模
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout, Activation
import keras
#语义网络
import collections
import networkx as nx  
import matplotlib.pyplot as plt
#LSA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# 将句子中词向量没有的词过滤掉
def RemoveWord_NOT_IN_w2v(comments,mark):
    model = Word2Vec.load(w2vModelFile)
    i=len(comments)-1
    while i>=0:
        out=[]
        for word in comments[i].split(' '):
            if word in model.wv.vocab:
                out.append(word)
        if len(out)==0:
            del comments[i]
            del mark[i]
        else:
            comments[i]=' '.join(out)
        i-=1
    return comments


# 建立词语到词向量的映射
def FormEmbedding(sentences):
    print('建立词语到词向量的映射')
    try:
        model = Word2Vec.load(w2vModelFile)
        # index2word词列表，vectors词向量数组
        w2v = dict(zip(model.wv.index2word, model.wv.vectors)) 
        w2index = {}  
        index = 1
        for sentence in sentences:
            for word in sentence.split(' '):
                if word not in w2index:
                    w2index[word] = index
                    index += 1
        print("文本中总共有{}个词" .format(len(w2index)))
        embeddings = np.zeros(shape=(len(w2index) + 1, 100), dtype=float)
        embeddings[0] = 0
        n_not_in_w2v = 0
        for word, index in w2index.items():
            if word in model.wv.vocab:
                embeddings[index] = w2v[word]
            else:
                print("not in w2v:",index,word)
                n_not_in_w2v += 1
        print("words not in w2v count: %d" % n_not_in_w2v)
        del model, w2v
        x = [[w2index[word] for word in sentence.split(' ')] for sentence in sentences]
    except:
        traceback.print_exc()
    return embeddings, x


# 切分训练集和测试集
def SplitTrainData(x,y,max_lenth):
    try:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
        # 超长的部分设置为0，截断
        x_train = sequence.pad_sequences(x_train, max_lenth)
        x_val = sequence.pad_sequences(x_val, max_lenth)
        # y弄成2分类，0负面，1正面
        y_train = keras.utils.to_categorical(y_train, num_classes=2)
        y_val = keras.utils.to_categorical(y_val, num_classes=2)
    except:
        print('切分训练集和测试集失败！')
        traceback.print_exc()
    finally:
        return x_train, x_val, y_train, y_val


# 建模
def BuildModel(posfile,negfile):
    try:
        #读取情感分析数据
        pos=LoadData(posfile)
        neg=LoadData(negfile)
        pos_mark=[1 for i in range(len(pos))]
        neg_mark=[0 for i in range(len(neg))]
        data=pos+neg
        mark=pos_mark+neg_mark
        #将句子中词向量没有的词过滤掉
        data=RemoveWord_NOT_IN_w2v(data,mark)
        #建立词语到词向量的映射
        embeddings,x=FormEmbedding(data)
        print('映射建立完毕')
        max_lenth=max([len(line) for line in x])
        print('max_length:',max_lenth)
        x_train,x_val,y_train,y_val=SplitTrainData(x,mark,max_lenth)
        print('Build model...')
        model = Sequential()
        model.add(Embedding(len(embeddings),len(embeddings[0]),input_length=max_lenth,weights=[embeddings],trainable=False,name='embeddings'))
        model.add(LSTM(1)) 
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('softmax'))
        #模型编译
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        #模型训练
        model.fit(x_train, y_train, batch_size=30)
        #模型预测评估
        pre = model.predict_classes(x_val,batch_size=30)
        score = model.evaluate(x_val,y_val,batch_size=30)
        print('pre:',pre)
        print('score:',score)
    except:
        print('建模失败！')
        traceback.print_exc()


# 语义网络
def yuyiwangluo(comments):
    num=20
    G=nx.Graph()
    plt.figure(figsize=(20,14))
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
    object_list=[]
    for line in comments:
        object_list+=line.split(' ')
    #词频统计
    word_counts = collections.Counter(object_list) # 对分词做词频统计
    word_counts_top = word_counts.most_common(num) # 获取最高频的词
    word = pd.DataFrame(word_counts_top, columns=['关键词','次数'])
    word_T = pd.DataFrame(word.values.T,columns=word.iloc[:,0])
    word_T.to_excel('./data/network/word_T.xls')
    net = pd.DataFrame(np.mat(np.zeros((num,num))),columns=word.iloc[:,0])
    net.to_excel('./data/network/net.xls')
    #构建语义关联矩阵
    k = 0
    word2=word
    word2_T = word_T
    word2_T.to_excel('./data/network/word2_T.xls')
    relation = list(0 for x in range(num))
    # 查看该段最高频的词是否在总的最高频的词列表中
    for j in range(num):
        for p in range(num):
            if word.iloc[j,0] == word2.iloc[p,0]:
                relation[j] = 1
                break
    #对于同段落内出现的最高频词，根据其出现次数加到语义关联矩阵的相应位置
    for j in range(num):
        if relation[j] == 1:
            for q in range(num):
                if relation[q] == 1:
                    net.iloc[j, q] = net.iloc[j, q] + word2_T.loc[1, word_T.iloc[0, q]]
    n = len(word)
    # 边的起点，终点，权重
    for i in range(n):
        for j in range(i, n):
            G.add_weighted_edges_from([(word.iloc[i, 0], word.iloc[j, 0], net.iloc[i, j])])
    
    nx.draw_networkx(G,pos=nx.spring_layout(G),node_color='white',edge_color='grey')
    plt.axis('off')
    plt.show()


# LSA
def LSAModel(comments):
    vectorizer = TfidfVectorizer(max_features =1000, max_df = 0.5,smooth_idf = True)
    X = vectorizer.fit_transform(comments)
    print('X.shape:',X.shape)
    #奇异分解
    svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)
    svd_model.fit(X)
    terms = vectorizer.get_feature_names()
    for i, comp in enumerate(svd_model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x:x[1], reverse=True)[:7]
        print("Topic "+str(i)+": ")
        for t in sorted_terms:
            print(t[0])
            print(" ")


# LDA
def LDAModel(posfile,negfile):
    neg = pd.read_csv(negfile,encoding="utf-8",header=None)
    pos = pd.read_csv(posfile,encoding="utf-8",header=None)
    #sep设置分割词，由于csv默认以半角逗号为分割词，而该词恰好在停词表中，因此，会导致读取错误
    #所以解决办法是手动设置一个不存在的分割词，如"tipdm",因为sep设置为多字符字符串，所以需要设置engine
    #engine取值为python或者c，c的速度快但是python更全面，一般取python
    stop = pd.read_csv(StopWordsFile,encoding="utf-8",header=None,sep="tipdm",engine="python")
    #停用词表转换为列表，加上空格和空字符串
    stop = [" ",""]+list(stop[0])
    # 将每一个评论中分词，转换为一个列表的列表项，通过空格分隔
    neg[1] = neg[0].apply(lambda s:s.split(" "))
    neg[2] = neg[1].apply(lambda x:[i for i in x if i not in stop])
    pos[1] = pos[0].apply(lambda s:s.split(" "))
    pos[2] = pos[1].apply(lambda x:[i for i in x if i not in stop])
    #导入LDA主题分析的人工智能模块
    from gensim import corpora,models
    #neg
    #制作词典:词语的向量化
    neg_dict = corpora.Dictionary(neg[2])
    #制作语料：将给给定语料（词典）转换为词袋模型
    neg_corpus = [neg_dict.doc2bow(i) for i in neg[2]]
    #LDA模型训练
    neg_lda = models.LdaModel(neg_corpus,num_topics=3,id2word=neg_dict)  
    #打印输出主题
    print('neg:',neg_lda.print_topics(num_topics=3))
    f = open("./data/LDA/negLDA.txt","w",encoding="utf-8")
    for i in range(3):
        f.write(neg_lda.print_topic(i))
        f.write("\n")
    f.close()
    #pos
    #制作词典:词语的向量化
    pos_dict = corpora.Dictionary(pos[2])
    #制作语料：将给给定语料（词典）转换为词袋模型
    pos_corpus = [pos_dict.doc2bow(i) for i in pos[2]]
    #LDA模型训练
    pos_lda = models.LdaModel(pos_corpus,num_topics=3,id2word=pos_dict)
    #打印输出主题
    print('pos:',pos_lda.print_topics(num_topics=3))
    f = open("./data/LDA/posLDA.txt","w",encoding="utf-8")
    for i in range(3):
        f.write(pos_lda.print_topic(i))
        f.write("\n")
    f.close()