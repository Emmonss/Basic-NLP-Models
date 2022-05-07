import os,pickle,sys
sys.path.append('../')
import pandas as pd
from tqdm import tqdm
from NER.DataProcess.data import WORD_COL, TAG_COL,SEM_SPLIT_SIGNAL,SEG_LEN
from NER.DataProcess.data_utils import *
from pprint import pprint
import numpy as np


SEM_SPLIT_SIGNAL =SEM_SPLIT_SIGNAL+[')',',',';','.','”']
def read_sor_tar_cropus(source_path,target_path):

    source_temp = []
    target_temp = []
    f_sor = open(source_path, 'r', encoding='utf-8')
    f_tar = open(target_path, 'r', encoding='utf-8')

    for item in f_sor:
        source_temp.append(item.strip())
    for item in f_tar:
        target_temp.append(item.strip())

    assert len(source_temp)==len(target_temp),"the souce-target len is not equal"

    return source_temp,target_temp

def trans_data(source_path,target_path,save_path,split_param=True):
    source_list = []
    target_list = []
    source_temp,target_temp = read_sor_tar_cropus(source_path,target_path)

    for index in tqdm(range(len(source_temp))):
        sor = source_temp[index].split()
        tar = target_temp[index].split()
        if len(sor) == len(tar):
            source_seg,tag_seg = split_long_paras_into_sentence(sor,
                                                                tar,
                                                                SEM_SPLIT_SIGNAL,
                                                                SEG_LEN)
            for index,sor in enumerate(source_seg):
                source_list.append(trans_sentence(sor))
                target_list.append(trans_sentence(tag_seg[index]))
    # pprint(source_list)
    # pprint(target_list)

    dict = {
        WORD_COL: source_list,
        TAG_COL: target_list
    }

    data = pd.DataFrame(dict)

    maxLen = max(len(row) for row in data[WORD_COL].values.tolist())
    print(maxLen)
    for index, row in enumerate(data[WORD_COL].values.tolist()):
        if len(row) == maxLen:
            print(index)
            print(row)

    data.to_csv(save_path, index=False)


if __name__ == '__main__':
    trans_data(source_path='../data/pd2014/source_BIO_2014_cropus.txt',
               target_path='../data/pd2014/target_BIO_2014_cropus.txt',
               save_path='../data/Proessdata/pd2014.csv')
    # trans_data(source_path='../data/pd2014/test_source.txt',
    #            target_path='../data/pd2014/test_target.txt',
    #            # save_path='../data/Proessdata/pd2014.csv')
    pass