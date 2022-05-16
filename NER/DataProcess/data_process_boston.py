#encoding:utf-8
import sys
sys.path.append('../')

import pandas as pd
from tqdm import tqdm
from NER.DataProcess.data_utils import *
from NER.DataProcess.data import WORD_COL,TAG_COL,SEM_SPLIT_SIGNAL,SEG_LEN
from pprint import pprint


SEM_SPLIT_SIGNAL =SEM_SPLIT_SIGNAL+['!','.','／','：']
def trans_data(data_path,save_path):
    word_list= []
    tag_list =[]
    with open(data_path,'r',encoding='utf-8') as fr:
        for sent in tqdm(fr):
            sent_seg = sent.strip().split("}}")
            sent_word = []
            sent_tag = []
            for item in sent_seg:
                if '{{' in item:
                    if item.index('{{') == 0:
                        tag = item.replace("{{", "").split(":")[0].strip().replace(" ","")
                        word = item.replace("{{", "").split(":")[1].strip().replace(" ","")
                        sent_tag.extend(get_bio(word, tag))
                        sent_word.extend(get_seg_word(word))

                    else:
                        item1 = item[:item.index('{{')].strip().replace(" ","")
                        item2 = item[item.index('{{'):].strip().replace(" ","")
                        sent_tag.extend([UNK_TAG] * len(item1))
                        sent_word.extend(get_seg_word(item1))

                        tag = item2.replace("{{", "").split(":")[0].strip().replace(" ","")
                        word = item2.replace("{{", "").split(":")[1].strip().replace(" ","")
                        sent_tag.extend(get_bio(word, tag))
                        sent_word.extend(get_seg_word(word))
                else:
                    item = item.strip().replace(" ","")
                    sent_tag.extend([UNK_TAG] * len(item))
                    sent_word.extend(get_seg_word(item))

            if len(sent_word) == len(sent_tag):
                source_seg, tag_seg = split_long_paras_into_sentence(sent_word,
                                                                     sent_tag,
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
    trans_data(data_path='../data/boson/origindata.txt',
               save_path='../data/Proessdata/boston.csv')
