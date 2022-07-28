import os,sys,re

from tqdm import tqdm
import numpy as np
import pandas as pd
from pprint import pprint
from tensorflow.keras.preprocessing.sequence import pad_sequences

'''
    在RNN中要控制每个文本的长度
    
    统一设计成BIO格式 UNK一律视为O
'''
UNK_TAG = "O"
PAD_TAG = "PAD"
END_TAG = "END"

UNK_WORD = "<unk>"
PAD_WORD = "<pad>"
END_WORD = '<end>'


SEM_SPLIT_SIGNAL = ['，','。','）','！','；','、','。','》','，',"？"]

WORD_COL = "words"
TAG_COL ="tags"
SEG_LEN = 30

def read_corpus(data):
    word_data = pd.read_csv(data)
    word_data[WORD_COL] = word_data[WORD_COL].apply(lambda x: x + " {}".format(END_WORD))
    word_data[TAG_COL] = word_data[TAG_COL].apply(lambda x: x + " {}".format(END_TAG))
    word_data.dropna(inplace=True)
    return word_data

def get_word_dict(word_data):
    wordIndexDict = {PAD_WORD: 0,
                     UNK_WORD: 1,
                     END_WORD: 2}
    wi = 3
    for row in tqdm(word_data[WORD_COL].values.tolist()):
        if type(row) == float:
            print(row)
            break
        for word in row.split(" "):
            if word not in wordIndexDict:
                wordIndexDict[word] = wi
                wi += 1
    vocabSize = wi
    maxLen = max(len(row) for row in word_data[WORD_COL].values.tolist())
    sequenceLengths = [len(row) for row in word_data[WORD_COL].values.tolist()]
    return wordIndexDict,vocabSize,maxLen,sequenceLengths


def get_tag_dict(word_data):
    # word_data[TAG_COL] = word_data[TAG_COL].apply(lambda x: re.sub("\-\S+", "", x))
    tag_set = set()
    tagIndexDict = {PAD_TAG: 0,
                    UNK_TAG: 1,
                    END_TAG: 2}
    ti = 3
    for row in tqdm(word_data[TAG_COL].values.tolist()):
        tags = row.strip().split(" ")
        for tag in tags:
            if tag not in tagIndexDict:
                tagIndexDict[tag] = ti
                ti += 1
            if tag.startswith('B') or tag.startswith('I'):
                tag_set.add(tag[2:])
    tagSum = len(list(tagIndexDict.keys()))

    return tagSum,tagIndexDict,tag_set

def word2index(wordIndexDict,word):
    if word in wordIndexDict.keys():
        return wordIndexDict[word]
    else:
        return wordIndexDict[UNK_WORD]

def tag2index(tagIndexDict,tag):
    if tag in tagIndexDict.keys():
        return tagIndexDict[tag]
    else:
        return tagIndexDict[UNK_TAG]


def get_words_label_data(path,
                         super_wordIndexDict=None,
                         super_tagIndexDict = None,
                         super_max_len=500,
                         val_flag=False,):
    word_data = read_corpus(path)

    wordIndexDict, vocabSize, maxLen, sequenceLengths = get_word_dict(word_data)
    tagSum, tagIndexDict,tag_set = get_tag_dict(word_data)

    if val_flag:
        maxLen = super_max_len
        wordIndexDict=super_wordIndexDict
        tagIndexDict = super_tagIndexDict
    word_data[WORD_COL] = word_data[WORD_COL].\
            apply(lambda x: [word2index(wordIndexDict,word) for word in x.split()])
    word_data[TAG_COL] = word_data[TAG_COL].\
            apply(lambda x: x.split() + [PAD_TAG for i in range(maxLen - len(x.split()))])
    word_data[TAG_COL] = word_data[TAG_COL].\
            apply(lambda x: [tag2index(tagIndexDict,tagItem) for tagItem in x])

    X = pad_sequences(word_data[WORD_COL], value=wordIndexDict[PAD_WORD],
                      padding='post', maxlen=maxLen)
    y = np.array(word_data[TAG_COL].values.tolist())

    return wordIndexDict,vocabSize,maxLen,sequenceLengths,tagSum,tagIndexDict,tag_set,X,y


if __name__ == '__main__':
    wordIndexDict,vocabSize,maxLen,sequenceLengths,tagSum,tagIndexDict,tag_set,\
    X,y = get_words_label_data('../data/Proessdata/weibo_train.csv')
    print(type(X))
    # pprint(tagIndexDict)
    # print(tag_set)
    # print(maxLen)
    #
    # print(np.shape(X))
    # print(np.shape(y))
    # print(X[1])
    # print(y[1])
    pass
    "saf"