import numpy as np
import math
from utils import Label2Tag,LoadMatrix

def getMaxPre(prob,Dic,ProDic):
    nextprob,maxpre = [],[]
    for i in range(len(Dic)):
        problist = []
        for j in range(len(prob)):
            problist.append(math.sqrt(prob[j]*Dic[i]*ProDic[j][i]))
        nextprob.append(max(problist))
        maxpre.append(problist.index(max(problist)))

    return nextprob,maxpre

def wordpred(word,Dic):
    if word in Dic:
        return Dic[word]
    else:
        return [0.0,0.0,0.0,1.0]

def Forward(sequence, Dic, ProDic):
    prob = np.zeros((len(sequence),4))
    maxpre = np.zeros((len(sequence)-1,4))
    prob[0] = wordpred(sequence[0],Dic)
    for i in range(1,len(sequence)):
        prob[i],maxpre[i-1] = getMaxPre(prob[i-1],wordpred(sequence[i],Dic),ProDic)

    lastmaxpro = prob[-1].tolist().index(max(prob[-1]))
    return lastmaxpro,maxpre

def Backward(lastmaxpro,maxpre):
    tag = []
    tag.append(Label2Tag(lastmaxpro))
    for i in range(len(maxpre)-1,-1,-1):
        lastmaxpro =(int)(maxpre[i].tolist()[lastmaxpro])
        tag.append(Label2Tag(lastmaxpro))
    tag.reverse()
    res = "".join(tag)
    return res

if __name__ == '__main__':
    Dic, ProDic = LoadMatrix()
    print(wordpred('蹶',Dic))
    print(wordpred('我',Dic))







