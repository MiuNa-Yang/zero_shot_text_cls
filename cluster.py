# -*- coding:utf-8 -*-
# @Time   : 2021/12/21 14:48
# @Author : xinhongyang
# @File   : cluster
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

with open('res/whitening_embeddings.pickle', 'rb') as f:
    sent_embdings = pickle.load(f)

with open("resources/新闻标题10000条.txt", 'r', encoding='utf-8') as f:
    lines = [line.replace("\u2022", "") for line in f.read().split("\n")]

k = 10
K = KMeans(n_clusters=k)
model = K.fit(sent_embdings)
labels = model.predict(sent_embdings)

centers = model.cluster_centers_


def cosine(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom


cosines = []
for cluster in range(k):
    tmp_coss = []
    for i in range(len(lines)):
        tmp_coss.append(cosine(centers[cluster], sent_embdings[i]))
    cosines.append(tmp_coss)

# cosines[k, i]表示第i个样本和第k个聚类中心的余弦相似度
for cluster in range(k):
    idx = np.argpartition(np.array(cosines[cluster]), -5)
#     print(len(idx))
    print("第{}组".format(cluster))
    print("例句: ")
    for i, id in enumerate(idx[-5:]):
        print("{}.".format(i + 1) + lines[id])
    print("-----------------------------------------------")
