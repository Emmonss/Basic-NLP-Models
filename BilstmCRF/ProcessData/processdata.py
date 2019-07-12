from tqdm import tqdm


def MakeTag(word):
    if(len(word)==1):
        return 'S'
    else:
        return 'B'+ 'M'*(len(word)-2)+'E'

def PData(filename,output):
    with open(filename,'r',encoding='utf-8') as fr:
        fw = open(output,'w',encoding='utf-8')
        for sentence in tqdm(fr):
            for word in sentence.split():
                tag = MakeTag(word)
                for i in range(len(tag)):
                    fw.write(word[i]+" "+tag[i]+"\n")
            fw.write("\n")
        fw.close()


if __name__ == '__main__':
    # filename = '../Data/msr_training.utf8'
    # output = 'train.utf8'
    filename = '../Data/msr_test_gold.utf8'
    output = 'test.utf8'
    PData(filename,output)