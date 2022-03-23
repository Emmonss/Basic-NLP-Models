from utils import Evaluation,LoadMatrix,MakeTag,makeseq
from HMMain import HMM
from tqdm import tqdm

def MakePred(filename):
    pred,true = [],[]
    Dic, ProDic = LoadMatrix()
    with open(filename,'r',encoding='utf-8') as fr:
        for seq in tqdm(fr):
            lpred,ltrue = [],[]
            for word in seq.split():
                truetag = MakeTag(word)
                for tag in truetag:
                    ltrue.append(tag)

            predtag,_ = HMM(makeseq(seq), Dic=Dic, ProDic=ProDic)
            for tag in predtag:
                lpred.append(tag)

            pred.append(lpred)
            true.append(ltrue)
        fr.close()
    return pred,true


def Evals(filename):
    pred,true = MakePred(filename)
    print(pred)
    print(true)
    # res = Evaluation(y_true=true,y_pred=pred)
    # print(res)

if __name__ == '__main__':
    filename = '../Data/msr_test_gold.utf8'
    Evals(filename)