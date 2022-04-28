import sys,os
sys.path.append('../')
import pandas as pd
import numpy as np
from Segment.Utils.basic_utils import load_pkl,save_pkl,reverse_dict
from Segment.DataProcess.data import WORD_COL,TAG_COL
from tqdm import tqdm
from pprint import pprint
import warnings

class Make_HMM_Matrx():
    def __init__(self):
        self.data = None
        self.file_path = None

        self.WordsAsTagDict = {}
        self.WordsAsTagDictProb = {}
        self.tag_dict = {}
        self.label2tag = {}

        self.tag_sum = 0
        self.TagsDistribution = None

        #这里默认是独立分词S，不然大概率会是B。分词任务中，不在字典的字就统一成S好了
        self.most_tag_name = None

        self.matrix_flag = False


    def make_with_train_file(self,file_path):
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)
        self._get_tags_num()
        self._get_dict_count()
        self._get_distribution()
        self._verify_all_matrix()

    def _verify_all_matrix(self):
        if self.tag_sum>0 \
                and len(self.WordsAsTagDict.keys())>0 \
                and len(self.WordsAsTagDictProb.keys())>0\
                and not self.TagsDistribution is None:
            self.matrix_flag = True
        else:
            warnings.warn("the HMM matrix is not well")

    def clear_all_matrix(self):
        self.matrix_flag = False
        pass

    def _get_distribution(self):
        self.TagsDistribution = np.zeros(shape=(self.tag_sum,self.tag_sum))

        tag_list = []
        for index,item in tqdm(self.data.iterrows()):
            tag_list.extend(item[TAG_COL].strip().split())
        for i in range(len(tag_list) - 1):
            cur = self.tag_dict[tag_list[i]]
            next = self.tag_dict[tag_list[i+1]]
            self.TagsDistribution[cur][next] += 1
        sum = np.sum(self.TagsDistribution, axis=1)
        self.TagsDistribution = (self.TagsDistribution.T / sum).T

        #替换一个极小值，让vertebi的时候求log or 求平方的时候不会报错
        self.TagsDistribution[np.where(self.TagsDistribution==0.0)] = 1e-8
        # pprint(self.TagsDistribution)

    def _get_dict_count(self):
        for index, item in tqdm(self.data.iterrows()):
            words = item[WORD_COL].strip().split()
            tags = item[TAG_COL].strip().split()
            for i in range(len(words)):
                if words[i] in self.WordsAsTagDict:
                    self.WordsAsTagDict[words[i]][self.tag_dict[tags[i]]] += 1
                else:
                    self.WordsAsTagDict[words[i]] = [0]*self.tag_sum
                    self.WordsAsTagDict[words[i]][self.tag_dict[tags[i]]] += 1

        for key in self.WordsAsTagDict:
            it = (np.array(self.WordsAsTagDict[key]) / np.sum(self.WordsAsTagDict[key])).tolist()
            self.WordsAsTagDictProb[key] = it
        # pprint(self.WordsAsTagDict)

        # pprint(self.WordsAsTagDictProb)

    def _get_tags_num(self):
        for index,item in tqdm(self.data.iterrows()):
            tags = item[TAG_COL].strip().split()
            for i in range(len(tags)):
                tag = tags[i]
                if tag in self.tag_dict.keys():
                    self.tag_dict[tag]+=1
                else:
                    self.tag_dict[tag]=0


        self.tag_sum = len(self.tag_dict.keys())

        for index,item in enumerate(self.tag_dict.keys()):
            self.tag_dict[item] = index

        self.label2tag = reverse_dict(self.tag_dict)
        #分词任务中就不求最频繁了默认为S 即未登录字设计未单个词
        self.most_tag_name = 'S'
        # most_tag_count = np.max(list(self.tag_dict.values()))
        # for item in self.tag_dict.keys():
        #     if self.tag_dict[item] == most_tag_count:
        #         self.most_tag_name = item


        # pprint(self.tag_dict)
        # pprint(self.label2tag)
        # print(self.most_tag_name,self.tag_sum)



    def load_from_path(self,model_path):
        load_dict = load_pkl(model_path)
        self.WordsAsTagDict = load_dict['WordsAsTagDict']
        self.WordsAsTagDictProb = load_dict['WordsAsTagDictProb']
        self.tag_dict = load_dict['tag_dict']
        self.label2tag = load_dict['label2tag']
        self.tag_sum = load_dict['tag_sum']
        self.TagsDistribution = load_dict['TagsDistribution']
        self.most_tag_name = load_dict['most_tag_name']
        self._verify_all_matrix()
        if self.matrix_flag:
            print('the hmm matrixs have been loaded successfully')
        else:
            warnings.warn('the hmm matrixs have some problems')

    def save_matrix(self,save_path,path_head):
        save_dict = {
            'WordsAsTagDict':self.WordsAsTagDict,
            'WordsAsTagDictProb':self.WordsAsTagDictProb,
            'tag_dict':self.tag_dict,
            'label2tag':self.label2tag,
            'tag_sum':self.tag_sum,
            'TagsDistribution':self.TagsDistribution,
            'most_tag_name':self.most_tag_name,
        }
        save_pkl(os.path.join(save_path,'{}_hmm.pkl'.format(path_head)),save_dict)

if __name__ == '__main__':
    # print([0]*4)
    h = Make_HMM_Matrx()
    file_name = '../datas/ProcessData/msr_train.csv'
    h.make_with_train_file(file_name)
    h.save_matrix(save_path='./matrixs',path_head='msr')

    # h.load_from_path(model_path='./matrixs/msr_hmm.pkl')
    pass

