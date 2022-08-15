'''
    词典制作，默认分词器为jieba分词默认版本
    默认的单字词典为bert里面默认的vocab.txt(当然也可以不用)
'''

import jieba
import os

from tqdm import tqdm


root_dict_path = '../processed_data/vocab.txt'
init_dict_flag = True
save_path=True

xiaohuangji_data_path = './xiaohuangji/data'
nlpcc_data_path = './nlpcc/data'

def read_cropus(path):
    res = []
    with open(path,'r',encoding='utf-8') as fr:
        for item in fr:
            res.append(item.strip())
    return res

def read_cropus_list(path_list):
    res = []
    for path in path_list:
        res.extend(read_cropus(path))
    return res

def make_dict(input_path_list):
    init_dict = []
    if init_dict_flag:
        init_dict = read_cropus(root_dict_path)
    print(init_dict[103:106])
    # sent_cuts = []
    # sents = read_cropus_list(input_path_list)
    # for sent in tqdm(sents):
    #     sent_cuts.extend(jieba.lcut(sent.strip()))
    # sent_cuts = list(set(sent_cuts))
    # init_dict.extend(sent_cuts)
    return init_dict

def write_dict(dict_list,path):
    assert isinstance(dict_list,list),"it should be a list type"



def get_xiaohuangji_dict():
    data_path_list = []
    for item in os.listdir(xiaohuangji_data_path):
        data_path_list.append(os.path.join(xiaohuangji_data_path,item))
    init_dict = make_dict(data_path_list)
    print(init_dict[:10])
    print(len(init_dict))

    pass

def get_nplcc_dict():
    pass

if __name__ == '__main__':
    get_xiaohuangji_dict()
    # print(os.path.abspath(os.path.join(xiaohuangji_data_path,'..')))
    pass