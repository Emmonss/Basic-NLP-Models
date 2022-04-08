

'''
针对pku的数据处理
    tag : B-M-E-S
原始数据：
    19980101-01-001-002/m  中共中央/nt  总书记/n  、/wu  国家/n  主席/n  江/nrf  泽民/nrg 
处理后数据(转成csv文件)：
    中共中央/nt  总书记/n  、/wu  国家/n  主席/n  江/nrf  泽民/nrg 
    S B E S S B E B E S B E S B E B E S 
'''

#切分长段落
def split_long_paras_into_sentence(word_list,seg_split_signal,seg_len):
    new_word_list = []
    pred,next = 0,0
    cur_len = 0
    for index,word in enumerate(word_list):
        cur_len+=len(word)
        if word in seg_split_signal:
            next = index
            if cur_len>seg_len:
                cur_len=0
                new_word_list.append(word_list[pred:next+1])
                pred = next+1
    new_word_list.append(word_list[pred:len(word_list)])
    return new_word_list


def get_single_item(word):
    if len(word)==1:
        return "S"
    elif len(word)>1:
        return "B "+"M "*(len(word)-2)+"E ".strip()
    return ""

def trans_sentence(item_list):
    res = ""
    if len(item_list)>0:
        for words in item_list:
            for word in words:
                res+="{} ".format(word)
    return res.strip()

def trans_tags(item_list):
    res = ""
    for word in item_list:
        res+=get_single_item(word)+" "
    return res
