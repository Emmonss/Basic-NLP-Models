

import jieba,re


def pre_token(text):
    return jieba.lcut(text)

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

def write_dict(dict_list,path):
    assert isinstance(dict_list,list),"it should be a list type"
    fw = open(path,'w',encoding='utf-8')
    for item in dict_list:
        if len(item)>0:
            fw.write(item.strip()+"\n")
    fw.close()


def is_string(item):
    return isinstance(item,str)

def do_lower_reg(item):
    reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
    return re.sub(reg, '', item).lower()

def get_max_from_list(item_list):
    return max(item_list)