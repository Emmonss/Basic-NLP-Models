'''
    词典制作，默认分词器为jieba分词默认版本
    默认的单字词典为bert里面默认的vocab.txt(当然也可以不用)
    更新：vocab.txt里面有大量BERT训练但是平时垃圾词汇，需要清理一下，只保留单字和符号和进位符
        清理后的初始单字字典为vocab_pure.txt
'''

import jieba
import os

from tqdm import tqdm


root_dict_path = '../processed_data/vocab_pure.txt'
init_dict_flag = True
save_path=True



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

def make_dict(input_path_list,max_word_len=6):
    init_dict = []
    if init_dict_flag:
        init_dict = read_cropus(root_dict_path)

    sent_words,f_sent_word = [],[]
    sents = read_cropus_list(input_path_list)
    for sent in tqdm(sents):
        words = jieba.lcut(sent.strip())
        sent_words.extend(words)
    sent_words = list(set(sent_words))

    #除去名字过长的，纯数字的分词
    for item in sent_words:
        if item not in init_dict and 0<len(item)<=max_word_len and not item.isdigit():
            f_sent_word.append(item)
    #保持[pad]之流在去重后还在最开始，后续的单字可以打乱
    i_d =init_dict[:7]
    a_d = init_dict[7:]
    a_d.extend(f_sent_word)
    a_d = list(set(a_d))
    print(a_d[:10])
    f_dict = i_d+a_d
    # print (f_dict[:10])
    return f_dict

def write_dict(dict_list,path):
    assert isinstance(dict_list,list),"it should be a list type"
    fw = open(path,'w',encoding='utf-8')
    for item in dict_list:
        if len(item)>0:
            fw.write(item.strip()+"\n")
    fw.close()


###############################################################################
def get_xiaohuangji_dict(in_path,out_path):
    print(in_path)
    data_path_list = []
    for item in os.listdir(in_path):
        data_path_list.append(os.path.join(in_path,item))
    dict_list = make_dict(data_path_list)
    write_dict(dict_list,out_path)
    pass

def get_nplcc_dict(in_path,out_path):
    print(in_path)
    data_path_list = []
    for item in os.listdir(in_path):
        data_path_list.append(os.path.join(in_path, item))
    dict_list = make_dict(data_path_list)
    print("dict_len：{}".format(len(dict_list)))
    write_dict(dict_list, out_path)
    pass

###############################################################################

if __name__ == '__main__':
    xiaohuangji_data_path = './xiaohuangji/data'
    nlpcc_data_path = './nlpcc/data'
    xiaohuangji_final_dict_path = '../processed_data/xhj_dict.txt'
    nlpcc_final_dict_path = '../processed_data/nlpcc_dict.txt'
    ###############################################################################
    get_xiaohuangji_dict(xiaohuangji_data_path,
                         xiaohuangji_final_dict_path)
    # get_nplcc_dict(nlpcc_data_path,
    #                      nlpcc_final_dict_path)
    pass