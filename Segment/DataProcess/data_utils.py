

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
    return res.strip()

'''
扬 帆 远 东 做 与 中 国 合 作 的 先 行,B M M E S S B E B E S B E
=========>
['扬帆远东', '做', '与', '中国', '合作', '的', '先行']
'''
def back_trans_sentence(sentence,tags):
    sent = sentence.strip().split()
    tag = tags.strip().split()
    res = []
    try:
        assert len(sent)==len(tag),"length is not equal"
        temp_word = ""
        c_index = 0
        for index,word in enumerate(sent):
            temp_word +=word
            t = tag[index]
            if t == 'S' or t=='E':
                res.append(temp_word)
                temp_word=""
                c_index = index
        if not c_index == len(sent):
            temp_word=''
            for index in range(c_index+1,len(sent)):
                temp_word+=sent[index]
            res.append(temp_word)
        return res
    except Exception as e:
        return sent


if __name__ == '__main__':
    sentence = "扬 帆 远 东 做 与 中 国 合 作 的 先 行"
    tags="B M M E S S B E B E B B M"
    res = back_trans_sentence(sentence,tags)
    print(res)
