
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from pprint import pprint
from BasicLayerModels.Transformer.backend.BertTokenizers import Tokenizer as myTokenizer
from SimSentence.BertOnly.SentsToken import read_corpus


class SimBertTokenizer:
    def __init__(self,vocab_path,
                 maxlen,
                 do_lower_case=True):
        self.maxLen=maxlen
        self.myTokenizer = myTokenizer(vocab_path,do_lower_case=do_lower_case)
        assert self.myTokenizer is not None,"myTokenizer object is None!"

    def SimSent2BertIndex(self, data,flag="train"):
        if flag=="train" or flag=="val":
            assert len(np.shape(data))==2 and np.shape(data)[1]==3,\
                "data should be like [[t11,t12,label1],...[tn1,tn2,labeln]]"
        elif flag=="predict":
            assert len(np.shape(data)) == 2 and (np.shape(data)[1] == 3 or np.shape(data)[1] == 2),\
                "data should be like [[t11,t12,label1],...[tn1,tn2,labeln]]"
        else:
            raise ValueError("flag should be any of [train,val,predict]")

        data = data[:self.maxLen]
        token_ids = []
        seg_ids = []
        tags = []
        for sent in tqdm(data):
            token_id_1, seg_id_1 = self._tokenize(sent[0])
            token_id_2, seg_id_2 = self._tokenize(sent[1])
            token_id_1.extend(token_id_2[1:])
            seg_id_1.extend([1] * len(seg_id_2[1:]))
            tags.append(str(sent[2]))
            token_ids.append(token_id_1)
            if flag == "train" or flag == "val":
                seg_ids.append(seg_id_1)
        token_ids = self._pad_seuqences(token_ids)
        if flag == "train" or flag == "val":
            seg_ids = self._pad_seuqences(seg_ids)

        #return
        if flag == "train" or flag == "val":
            return np.array(token_ids), np.array(seg_ids), np.array(tags).astype(np.int32)
        else:
            return np.array(token_ids), np.array(seg_ids)
    def _tokenize(self,text):
        token_id, seg_id = self.myTokenizer.encode(text)
        return token_id,seg_id

    def _pad_seuqences(self,tokens):
        return tf.keras.preprocessing.\
            sequence.pad_sequences(tokens,maxlen=self.maxLen,truncating='post',padding='post')



if __name__ == '__main__':
    vocab_path = '../../BasicLayerModels/modelHub/chinese_L-12_H-768_A-12/vocab.txt'
    tokenizer = SimBertTokenizer(vocab_path,
                                 maxlen=100,
                                 do_lower_case=True)
    # data =[
    #     ["你是傻逼","我是傻逼",1],
    #     ["他是傻逼","我们都是傻逼",1],
    #     ["他是傻逼", "我们都是傻逼", 1]
    # ]
    data = read_corpus('../datas/bq_corpus/train.tsv')[:10]
    pprint(data)
    token_ids,seg_ids,tags = tokenizer.SimSent2BertIndex(data)
    print("token_ids".center(30,'='))
    pprint(token_ids)
    print("seg_ids".center(30, '='))
    pprint(seg_ids)
    print("tags".center(30, '='))
    pprint(tags)