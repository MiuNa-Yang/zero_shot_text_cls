# _*_ coding:utf-8 _*_
# @Time   : 2021/12/21 14:03
# @Author : xinhongyang
# @File   : sent_embeddings
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import os
import datetime
import re
from tqdm import tqdm
import pickle


def sent_to_vec(sen, tokenizer, bert_layer, pooling='last_avg', maxlen=128):
    """
    将文本数据转化
    :param bert_layer: bert层
    :param tokenizer: 分词器
    :param sen: 输入句子
    :param pooling: 输出方法
    :param maxlen: 最大长度
    :return: 裸bert出来的句向量
    """

    with torch.no_grad():
        inputs = tokenizer(sen, return_tensors="pt", padding=True, truncation=True, max_length=maxlen)
        inputs['input_ids'] = inputs['input_ids']
        inputs['token_type_ids'] = inputs['token_type_ids']
        inputs['attention_mask'] = inputs['attention_mask']
        # print(inputs)
        hidden_states = bert_layer(**inputs, return_dict=True, output_hidden_states=True).hidden_states

        if pooling == 'first_last_avg':
            output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
        elif pooling == 'last_avg':
            output_hidden_state = (hidden_states[-1]).mean(dim=1)
        elif pooling == 'last2avg':
            output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
        elif pooling == 'cls':
            output_hidden_state = (hidden_states[-1])[:, 0, :]
        else:
            raise Exception("unknown pooling {}".format(pooling))
        # output_hidden_state = (hidden_states[-1]).mean(dim=1)
        vec = output_hidden_state.cpu().numpy()[0]

    return vec


def sents_to_vecs(sents, tokenizer, bert_layer, pooling='last_avg', maxlen=512):
    vecs = []

    for sent in sents:
        vec = sent_to_vec(sent, tokenizer, bert_layer, pooling=pooling, maxlen=maxlen)
        vecs.append(vec)

    vecs = np.array(vecs)

    return vecs


def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    # vecs = np.concatenate(vecs, axis=0)
    # print(vecs.shape)
    mu = vecs.mean(axis=0, keepdims=True)
    # print(mu.shape)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    # return None, None
    # return W, -mu
    return W[:, :256], -mu


def normalize(vecs):
    """标准化
    """
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5


def transform_and_normalize(vecs, kernel, bias):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return normalize(vecs)


if __name__ == "__main__":

    BERT_PATH = 'pretrained_model/roberta'

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    bert_layer = BertModel.from_pretrained(BERT_PATH)

    with open("resources/新闻标题10000条.txt", 'r', encoding='utf-8') as f:
        lines = [line.replace("\u2022", "") for line in f.read().split("\n")]

    # # print(len(lines))
    # print(lines)

    print('data loaded')
    vecs = sents_to_vecs(lines, pooling="first_last_avg", tokenizer=tokenizer, bert_layer=bert_layer)
    print('vecs get')
    kernel, bias = compute_kernel_bias(vecs)
    print('kernel get')
    whitening_embeddings = transform_and_normalize(vecs, kernel, bias)
    print('embeddings get')
    # print(lines, len(vecs), len(whitening_embeddings), len(vecs[0]), len(whitening_embeddings[0]))
    with open('res/whitening_embeddings.pickle', 'wb') as f:
        pickle.dump(whitening_embeddings, f)
