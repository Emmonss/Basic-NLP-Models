from pprint import pprint

def read_corpus(corpus_path,flag="train",seg='\t'):
    assert flag in ['train','val','predict'],'flag should be any of [train,val,predict]'
    res =[]
    with open(corpus_path,'r',encoding='utf-8') as fr:
        for item in fr:
            seg = item.strip().split('\t')
            if (flag=='train' or flag=='val') and len(seg)==3:
                res.append(seg)
            elif(flag=='predict' ) and len(seg)==2:
                res.append(seg)

    return res


if __name__ == '__main__':
    read_corpus(corpus_path='../datas/bq_corpus/train.tsv')


