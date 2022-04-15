import os,pickle
import pandas as pd
from tqdm import tqdm
from Segment.DataProcess.data import WORD_COL, TAG_COL,SEM_SPLIT_SIGNAL
from Segment.DataProcess.data_utils import *
'''

针对pku的数据处理
    tag : B-M-E-S
原始数据：
    19980101-01-001-002/m  中共中央/nt  总书记/n  、/wu  国家/n  主席/n  江/nrf  泽民/nrg 
处理后数据(转成csv文件)：
    中共中央/nt  总书记/n  、/wu  国家/n  主席/n  江/nrf  泽民/nrg 
    S B E S S B E B E S B E S B E B E S 
'''

SEM_SPLIT_SIGNAL_PKU = SEM_SPLIT_SIGNAL+\
    ['山西省','辽宁省','吉林省','黑龙江省','江苏省','山东省','福建省','江西省'
     '安徽省','河北省','甘肃省','浙江省','河南省','湖南省','广西省',
     '四川省','云南省','贵州省','陕西省','湖北省','重庆市','广东省']+\
    ['县长','所长','院长','处长','部长','总经理','宣传部','科学院','医院','委员会']
SEGMENT_MAX_SENTENCE_LEN_PKU = 100
MAX_SENTEN_LEN = 500

def trans_data(path,save_path,split_param = True):
    words = []
    tags = []
    dict ={}
    with open(path,'r',encoding='utf-8') as fr:
        for item in tqdm(fr):
            temp_list = item.strip().split()
            item_list = []
            if len(temp_list)>1:
                for word in temp_list[1:]:
                    item_list.append(word.split('/')[0].split('{')[0].replace("[","").strip())
                if split_param:
                    item_list = split_long_paras_into_sentence(item_list,SEM_SPLIT_SIGNAL_PKU,SEGMENT_MAX_SENTENCE_LEN_PKU)
                    # print(item_list)
                    for item_l in item_list:
                        temp = trans_sentence(item_l)
                        if len(item_l) > 1 and len(temp)<=MAX_SENTEN_LEN:
                            tags.append(trans_tags(item_l))
                            words.append(trans_sentence(item_l))
                else:
                    temp = trans_sentence(item_list)
                    if len(item_list) > 1 and len(temp) <= MAX_SENTEN_LEN:
                        tags.append(trans_tags(item_list))
                        words.append(trans_sentence(item_list))
        dict = {
            WORD_COL:words,
            TAG_COL:tags
        }
    #
    data=pd.DataFrame(dict)

    maxLen = max(len(row) for row in data[WORD_COL].values.tolist())
    print(maxLen)
    # for index,row in enumerate(data[WORD_COL].values.tolist()):
    #     if len(row)==maxLen:
    #         print(index)
    #         print(row)
    # pprint(maxLen)

    data.to_csv(save_path,index=False)

if __name__ == '__main__':
    pku_train_path = '../datas/msr_data/pku.txt'

    pku_train_csv = '../datas/ProcessData/pku_data.csv'

    trans_data(pku_train_path,pku_train_csv)
