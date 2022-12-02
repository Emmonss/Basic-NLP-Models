

import jieba,re
from ChatBot import train_config

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

def make_dict(input_list,output_list,pretoken=False,
              init_dict = ['[PAD]', '[STA]','[END]','[UNK]']):
    if isinstance(input_list,str):
        input_list = [input_list]
    if not isinstance(init_dict,list):
        raise ValueError("the init dict should be a list")
    dict_list = []
    for item in input_list:
        with open(item,'r',encoding='utf-8') as fr:
            for sent in fr:
                if pretoken:
                    dict_list.extend(pretoken(sent.strip()))
                else:
                    for word in sent.strip():
                        dict_list.append(word)
    dict_list = list(set(dict_list))
    init_dict.extend(dict_list)
    print("dict len:{}".format(len(init_dict)))
    with open(output_list,'w',encoding='utf-8') as fw:
        for word in init_dict:
            fw.write("{}\n".format(word))
    fw.close()

if __name__ == '__main__':
    inpath = train_config.inpath_xhj
    tarpath = train_config.tarpath_xhj
    dictpath = train_config.dict_path
    make_dict([inpath,tarpath],dictpath)
    pass