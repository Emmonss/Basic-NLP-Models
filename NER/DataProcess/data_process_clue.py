#encoding:utf-8
import sys
sys.path.append('../')

import pandas as pd
from tqdm import tqdm
from NER.DataProcess.data_utils import *
from NER.DataProcess.data import WORD_COL,TAG_COL,SEM_SPLIT_SIGNAL,SEG_LEN
from pprint import pprint
import json

SEM_SPLIT_SIGNAL =SEM_SPLIT_SIGNAL+['!','.','／','：']
def trans_data(data_path,save_path):
    word_list = []
    tag_list = []
    with open(data_path, 'r', encoding='utf-8') as fr:
        for line in tqdm(fr):
            line_dict = json.loads(line.strip())
            text = line_dict['text']
            word_tag = [UNK_TAG] * len(text)
            labels = line_dict['label']
            for tag in labels.keys():
                for tag_word in labels[tag]:
                    tag_f = get_bio(tag_word, tag)
                    for index_set in labels[tag][tag_word]:
                        word_tag[index_set[0]:index_set[1] + 1] = tag_f

            word_list.append(trans_sentence(text))
            tag_list.append(trans_sentence(word_tag))

        dict = {
            WORD_COL: word_list,
            TAG_COL: tag_list
        }

        data = pd.DataFrame(dict)

        maxLen = max(len(row.split()) for row in data[WORD_COL].values.tolist())
        print(maxLen)
        # for index, row in enumerate(data[WORD_COL].values.tolist()):
        #     if len(row.split()) == maxLen:
        #         print(index)
        #         print(row)

        data.to_csv(save_path, index=False)

if __name__ == '__main__':
    trans_data(data_path='../data/clue/test.json',
               save_path='../data/Proessdata/clue_test.csv')

