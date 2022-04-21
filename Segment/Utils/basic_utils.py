import pickle,sys
sys.path.append('../')

from Segment.DataProcess.data import END_TAG,END_WORD,PAD_TAG,PAD_WORD

def save_pkl(path,obj):
    with open(path,'wb') as fw:
        pickle.dump(obj,fw)
    fw.close()

def load_pkl(path):
    with open(path,'rb') as fr:
        data = pickle.load(fr)
    return data

def reverse_dict(init_dict):
    res = {}
    for key,value in init_dict.items():
        res[value] = key
    return res

def index2tag(index_list,tag_dict,len_word):
    res = ''
    count = 0
    for item in index_list:
        # if not (tag_dict[item] == END_TAG or tag_dict[item] == PAD_TAG):
        if count<len_word:
            count+=1
            res+=tag_dict[item]+' '
    return res.strip()

def index2word(index_list,word_dict):
    res = ''
    for item in index_list:
        if not (word_dict[item] == END_WORD or word_dict[item] == PAD_WORD):
            res += word_dict[item] + ' '
    return res.strip()