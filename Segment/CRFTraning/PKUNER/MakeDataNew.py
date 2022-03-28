import os
import random
import pickle


def openfile(filedir,tr_pkl = 'train.pkl', te_pkl = 'test.pkl'):
    seq = []
    text = os.listdir(filedir)
    for t in text:
        path = os.path.join(filedir,t)
        with open(path,'r',encoding='utf-8') as fr:
            for item in fr:
                if (len(item) > 1):
                    seg = []
                    item = item.split()[1:]
                    for word in item:
                        it = []
                        it.append(word.split('/')[0])
                        it.append(word.split('/')[1])
                        seg.append(it)
                    seq.append(seg)

    random.shuffle(seq)
    k = int(len(seq)/10)
    train_set = seq[0:k*9]
    test_set = seq[k*9:len(seq)]

    with open(tr_pkl, 'wb') as fw:
        pickle.dump(train_set, fw)
    fw.close()

    with open(te_pkl, 'wb') as fw:
        pickle.dump(train_set, fw)
    fw.close()

    return train_set,test_set

if __name__ == '__main__':
    dir = '../../datas/corpus_data'
    train_set, test_set = openfile(dir)
    print(test_set[0])