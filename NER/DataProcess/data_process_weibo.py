#encoding:utf-8
import sys
sys.path.append('../')

import pandas as pd
from tqdm import tqdm
from NER.DataProcess.data_utils import *
from NER.DataProcess.data import WORD_COL,TAG_COL,SEM_SPLIT_SIGNAL,SEG_LEN
from pprint import pprint


SEM_SPLIT_SIGNAL =SEM_SPLIT_SIGNAL+['@']

def trans_data(data_path,save_path):
    with open(data_path, 'r', encoding='utf-8') as fr:
        word_list = []
        tag_list = []
        sent = []
        tag = []
        for item in tqdm(fr):
            try:
                if len(item.split()) > 0:
                    sent.append(item.split()[0].strip())
                    tag.append(item.split()[1].strip())
                else:
                    if len(sent) == len(tag) and len(sent) > 0:
                        source_seg, tag_seg = split_long_paras_into_sentence(sent,
                                                                             tag,
                                                                             SEM_SPLIT_SIGNAL,
                                                                             SEG_LEN)
                        for index, sor in enumerate(source_seg):
                            word_list.append(trans_sentence(sor))
                            tag_list.append(trans_sentence(tag_seg[index]))
                    sent = []
                    tag = []
            except:
                continue
        if len(sent) == len(tag) and len(sent) > 0:
            source_seg, tag_seg = split_long_paras_into_sentence(sent,
                                                                 tag,
                                                                 SEM_SPLIT_SIGNAL,
                                                                 SEG_LEN)
            for index, sor in enumerate(source_seg):
                word_list.append(trans_sentence(sor))
                tag_list.append(trans_sentence(tag_seg[index]))


        dict = {
            WORD_COL: word_list,
            TAG_COL: tag_list
        }

        data = pd.DataFrame(dict)

        maxLen = max(len(row.split()) for row in data[WORD_COL].values.tolist())
        print(maxLen)
        for index, row in enumerate(data[WORD_COL].values.tolist()):
            if len(row.split()) == maxLen:
                print(index)
                print(row)

        data.to_csv(save_path, index=False)



if __name__ == '__main__':
    # trans_data(data_path='../data/weibo/train.txt',
    #            save_path='../data/Proessdata/weibo_train.csv')

    # trans_data(data_path='../data/weibo/val.txt',
    #            save_path='../data/Proessdata/weibo_val.csv')

    trans_data(data_path='../data/weibo/test.txt',
               save_path='../data/Proessdata/weibo_test.csv')


