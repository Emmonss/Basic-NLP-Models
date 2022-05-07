import os,sys,re

from tqdm import tqdm
import numpy as np
import pandas as pd

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
    word_data[TAG_COL] = word_data[TAG_COL].apply(lambda x: re.sub("\-\S+", "", x))

    tagIndexDict = {PAD_TAG: 0,
                    UNK_TAG: 1,
                    END_TAG: 2}
    ti = 3
    for row in tqdm(word_data[TAG_COL].values.tolist()):
        for tag in row.split(" "):
            if tag not in tagIndexDict:
                tagIndexDict[tag] = ti
                ti += 1
    tagSum = len(list(tagIndexDict.keys()))

    return tagSum,tagIndexDict

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
    pass

if __name__ == '__main__':
    pass