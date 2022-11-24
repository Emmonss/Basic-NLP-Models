import sys,os,math
sys.path.append('../')

import numpy as np
from Segment.HMM.HMM_Matrix import Make_HMM_Matrx

def segment(sent,tag):
    res = ''
    for i in range(len(sent)):
        if tag[i] == 'S' or tag[i] == 'E':
            res+=sent[i]+" "
        else:
            res+=sent[i]
    return res

class Vertebi():
    def __init__(self,mode='train',model_path=None,data_path = None):
        self.mode= mode
        self.model_path = model_path
        self.data_path = data_path
        self.hmm = Make_HMM_Matrx()
        self._init_hmm_matrix()


    def _init_hmm_matrix(self):
        if self.mode=='train' and not self.data_path==None:
            self.hmm.make_with_train_file(self.data_path)
        elif self.mode=='load' and not self.model_path==None:
            self.hmm.load_from_path(self.model_path)
        else:
            assert "the mode is not vetify"
        assert self.hmm.matrix_flag,"the hmm matrixs have not been corrtected established"

    def _getMaxPre(self,prob,Dic,ProDic,mode = 'sqrt'):
        nextprob, maxpre = [], []
        for i in range(len(Dic)):
            problist = []
            for j in range(len(prob)):
                if mode == 'sqrt':
                    problist.append(math.sqrt(prob[j] * Dic[i] * ProDic[j][i]))
                elif mode== 'log':
                    problist.append(math.log(prob[j] * Dic[i] * ProDic[j][i]))
                else:
                    problist.append(math.sqrt(prob[j] * Dic[i] * ProDic[j][i]))
            nextprob.append(max(problist))
            maxpre.append(problist.index(max(problist)))

        return nextprob, maxpre

    def _wordpred(self,word, Dic):
        if word in Dic:
            return Dic[word]
        else:
            prob = [0.0]*self.hmm.tag_sum
            prob[self.hmm.tag_dict[self.hmm.most_tag_name]]=1.0
            return prob
    def _forward(self,sequence):
        prob = np.zeros((len(sequence), self.hmm.tag_sum))
        maxpre = np.zeros((len(sequence) - 1, self.hmm.tag_sum))
        prob[0] = self._wordpred(sequence[0], self.hmm.WordsAsTagDictProb)
        for i in range(1, len(sequence)):
            prob[i], maxpre[i - 1] = self._getMaxPre(prob[i - 1],
                                   self._wordpred(sequence[i], self.hmm.WordsAsTagDictProb),
                                   self.hmm.TagsDistribution)

        lastmaxpro = prob[-1].tolist().index(max(prob[-1]))
        return lastmaxpro, maxpre

    def _backword(self,lastmaxpro,maxpre):
        tag = []
        tag.append(self.hmm.label2tag[lastmaxpro])
        for i in range(len(maxpre) - 1, -1, -1):
            lastmaxpro = (int)(maxpre[i].tolist()[lastmaxpro])
            tag.append(self.hmm.label2tag[lastmaxpro])
        tag.reverse()
        res = "".join(tag)
        return res

    def get_segment_tags(self,words):
        lastmaxpro, maxpre = self._forward(words)
        tag = self._backword(lastmaxpro, maxpre)
        # return tag
        res = segment(sequence,tag)
        return tag,res

if __name__ == '__main__':
    sequence = '“北京健康宝”是一个方便个人查询自身防疫相关健康状态的小程序，所有在京及进（返）京人员均可使用。通过百度APP、微信、支付宝，'
    v = Vertebi(mode='load',model_path='./matrixs/msr_hmm.pkl')
    tag,res = v.get_segment_tags(sequence)
    print(sequence)
    print(tag)
    print(res)