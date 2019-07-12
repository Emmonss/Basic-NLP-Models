from Vertebi import Forward,Backward
from utils import segment,LoadMatrix


def HMM(sequence,Dic,ProDic):
    lastmaxpro, maxpre = Forward(sequence, Dic=Dic, ProDic= ProDic)
    tag = Backward(lastmaxpro,maxpre)
    res = segment(sequence,tag)
    return tag,res



if __name__ == '__main__':
    sequence ='希腊的蹶经济结构较特殊。'

    Dic, ProDic = LoadMatrix()
    tag, res = HMM(sequence, Dic, ProDic)
    print(res)