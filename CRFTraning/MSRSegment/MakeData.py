
#设计tag
def MakeTag(word):
    if(len(word)==1):
        return 'S'
    else:
        return 'B'+ 'M'*(len(word)-2)+'E'

#加载数据
def LoadData(filename):
    seq = []
    with open(filename,'r',encoding='utf-8') as fr:
         for item in fr:
             res = []
             words = item.split()
             for word in words:
                tag = MakeTag(word)
                for i in range(len(tag)):
                    it = []
                    it.append(word[i])
                    it.append(tag[i])
                    res.append(it)
             seq.append(res)
    return  seq

def MakeOneNode(sent):
    res = []
    for word in sent:
        res.append(word)
    return res


if __name__ == '__main__':
    filename = '../Data/msr_training.utf8'
    res = LoadData(filename=filename)
    print(res[0])
    # print(res[1])
    # print(res[2])
    # print(MakeTag("你是"))