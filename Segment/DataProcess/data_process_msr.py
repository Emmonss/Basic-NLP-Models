import os,pickle
import pandas as pd
from tqdm import tqdm
from Segment.DataProcess.data import WORD_COL, TAG_COL,SEM_SPLIT_SIGNAL
from Segment.DataProcess.data_utils import *
'''
针对msr的数据处理
    tag : B-M-E-S
原始数据：
    这  首先  是  个  民族  问题  ，  民族  的  感情  问题  。
处理后数据(转成csv文件)：
    这 首 先 是 个 民 族 问 题 民 族 的 感 情 问 题 。
    S B E S S B E B E S B E S B E B E S 
'''

SEM_SPLIT_SIGNAL_MSR = SEM_SPLIT_SIGNAL+["多云"]
SEGMENT_MAX_SENTENCE_LEN_MSR = 100
MAX_SENTEN_LEN = 500

def trans_data(path,save_path,split_param=True):
    words = []
    tags = []
    dict ={}
    with open(path,'r',encoding='utf-8') as fr:
        for item in tqdm(fr):
            item_list = item.strip().split()
            if len(item_list)>1:
                if split_param:
                    item_list = split_long_paras_into_sentence(item_list,SEM_SPLIT_SIGNAL_MSR,SEGMENT_MAX_SENTENCE_LEN_MSR)
                    for item_l in item_list:
                        temp = trans_sentence(item_l)
                        if len(item_l) > 1 and len(temp)<=MAX_SENTEN_LEN:
                            tags.append(trans_tags(item_l))
                            words.append(trans_sentence(item_l))
                else:
                    temp = trans_sentence(item_list)
                    if len(temp)<=MAX_SENTEN_LEN:
                        tags.append(trans_tags(item_list))
                        words.append(trans_sentence(item_list))
        dict = {
            WORD_COL:words,
            TAG_COL:tags
        }

    data=pd.DataFrame(dict)

    maxLen = max(len(row) for row in data[WORD_COL].values.tolist())
    print(maxLen)
    # for index, row in enumerate(data[WORD_COL].values.tolist()):
    #     if len(row) == maxLen:
    #         print(index)
    #         print(row)

    data.to_csv(save_path,index=False)

if __name__ == '__main__':
    msr_train_path = '../datas/msr_data/msr_training.utf8'
    msr_test_path = '../datas/msr_data/msr_test_gold.utf8'

    msr_train_csv = '../datas/ProcessData/msr_train.csv'
    msr_test_csv = '../datas/ProcessData/msr_test.csv'

    trans_data(msr_train_path,msr_train_csv)
    trans_data(msr_test_path, msr_test_csv)