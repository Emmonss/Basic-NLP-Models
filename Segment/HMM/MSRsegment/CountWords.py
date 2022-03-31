import numpy as np
import pickle
from utils import Tag2label,MakeTag
from tqdm import tqdm

Dic = {}
ProDic = {}

def Count(word, tag):
    assert len(word) == len(tag),"单词与标注不符"

    for i in range(len(word)):
        if word[i] in Dic:
            Dic[word[i]][Tag2label(tag[i])] +=1
        else:
            Dic[word[i]] = [0,0,0,0]
            Dic[word[i]][Tag2label(tag[i])] += 1


def MakePro(filename):
    with open(filename,'r',encoding='utf-8') as fr:
        for item in tqdm(fr):
            words = item.split()
            for word in words:
                tag = MakeTag(word)
                Count(word, tag)
        fr.close()

    for key in Dic:
        it = (np.array(Dic[key]) /np.sum(Dic[key])).tolist()
        ProDic[key] = it

    with open('Dic.pkl', 'wb') as fw:
        pickle.dump(Dic, fw)
    fw.close()

    with open('ProDic.pkl', 'wb') as fw:
        pickle.dump(ProDic, fw)
    fw.close()


def MakeDistribution(filename):
    Dis = np.zeros(shape=(4,4))

    with open(filename,'r',encoding='utf-8') as fr:
        for item in tqdm(fr):
            seq = ''
            words = item.split()
            for word in words:
                tag = MakeTag(word)
                seq +=tag
            for i in range(len(seq)-1):
                Dis[Tag2label(seq[i])][Tag2label(seq[i+1])]+=1


        sum = np.sum(Dis,axis=1)
        Dis = (Dis.T/sum).T
        print(Dis)
        with open('Distribution.pkl', 'wb') as fw:
            pickle.dump(Dis, fw)


if __name__ == '__main__':
    filename = '../msr_data/msr_training.utf8'
    # MakePro(filename)
    MakeDistribution(filename)




