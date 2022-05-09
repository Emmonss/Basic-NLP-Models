import sys
sys.path.append('../')
from NER.DataProcess.data import UNK_TAG

# 切分长段落
def split_long_paras_into_sentence(word_list, tag_list,seg_split_signal, seg_len):
    new_word_list = []
    new_tag_list = []
    pred, next = 0, 0
    cur_len = 0
    for index, word in enumerate(word_list):
        cur_len += len(word)
        if word in seg_split_signal and tag_list[index]==UNK_TAG:
            next = index
            if cur_len > seg_len:
                cur_len = 0
                if len(word_list)-(next + 1)>10:
                    new_word_list.append(word_list[pred:next + 1])
                    new_tag_list.append(tag_list[pred:next + 1])
                    pred = next + 1
                else:
                    #字段末尾距离结束小于10时，直接截取到末尾停止
                    new_word_list.append(word_list[pred:len(word_list)])
                    new_tag_list.append(tag_list[pred:len(word_list)])
                    pred = len(word_list)
                    break
    if pred<len(word_list):
        new_word_list.append(word_list[pred:len(word_list)])
        new_tag_list.append(tag_list[pred:len(word_list)])
    return new_word_list,new_tag_list

def trans_sentence(item_list):
    res = ""
    if len(item_list)>0:
        for words in item_list:
            res+="{} ".format(words)
    return res.strip()

#shu
def get_bio(text,tag):
    if tag==UNK_TAG:
        return [UNK_TAG]*len(text)
    else:
        return ['B_{}'.format(tag)]+['I_{}'.format(tag)]*(len(text)-1)


if __name__ == '__main__':
    print(get_bio("中共中央","O"))