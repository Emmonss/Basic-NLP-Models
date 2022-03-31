import random
import pickle


#加载数据
def LoadData(filename,tr_pkl = 'train.pkl', te_pkl = 'test.pkl'):
    seq = []
    with open(filename,'r',encoding='utf-8') as fr:
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



def MakeOneNode(sent):
    res = []
    for word in sent:
        res.append(word)
    return res


if __name__ == '__main__':
    filename = '../../datas/msr_data/pku.txt'
    train_set, test_set = LoadData(filename=filename)
    print(test_set[0])
    # print(res[1])
    # print(res[2])
    # print(MakeTag("你是"))