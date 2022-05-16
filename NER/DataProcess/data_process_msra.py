import sys
sys.path.append('../')

import pandas as pd
from tqdm import tqdm
from NER.DataProcess.data_utils import *
from NER.DataProcess.data import WORD_COL,TAG_COL,SEM_SPLIT_SIGNAL,SEG_LEN
from pprint import pprint


SEM_SPLIT_SIGNAL =SEM_SPLIT_SIGNAL+['℃']
def trans_data(data_path,save_path):
    word_list= []
    tag_list =[]
    with open(data_path,'r',encoding='utf-8') as fr:
        for sent in tqdm(fr):
            sent_word = []
            sent_tag = []
            word_tags = sent.strip().split()
            for item in word_tags:
                word = item.split('/')[0]
                tag = item.split('/')[1]
                sent_tag.extend(get_bio(word,tag))
                sent_word.extend(get_seg_word(word))
            if len(sent_word) == len(sent_tag):
                source_seg, tag_seg = split_long_paras_into_sentence(sent_word,
                                                                     sent_tag,
                                                                     SEM_SPLIT_SIGNAL,
                                                                     SEG_LEN)
                for index, sor in enumerate(source_seg):
                    word_list.append(trans_sentence(sor))
                    tag_list.append(trans_sentence(tag_seg[index]))
    #
    # pprint(word_list)
    # pprint(tag_list)
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
    trans_data(data_path='../data/msra/test_label.txt',
               save_path='../data/Proessdata/msra_test.csv')
    # trans_data(data_path='../data/msra/train.txt',
    #            save_path='../data/Proessdata/msra_train.csv')
    pass