import numpy as np
import pandas as pd
import re,sys
sys.path.append('../../')
sys.path.append('../')

from NER.Utils.basic_utils import load_pkl
from tqdm import tqdm
from pprint import pprint
from NER.DataProcess.data import TAG_COL,UNK_TAG,WORD_COL

TP=0
FP=1
FN=2
PRECISION=3
RECALL=4
F1_SCORE=5

EVAL_ITEM_LEN=6


def get_ner_word(word_list,tag_list):
    res_list = []
    ner_word = ''
    for index in range(len(word_list)):
        if not tag_list[index]==UNK_TAG:
            ner_word+=word_list[index]
        else:
            if not ner_word=='':
                res_list.append(ner_word)
            ner_word=''
    if not ner_word == '':
        res_list.append(ner_word)

    return res_list


def get_pred_main(true_csv,pred_csv,ner_dict,mid='-'):
    assert len(pred_csv)==len(true_csv),'length is not equal'
    eval_dict = {}
    preds,all = 0,0
    for item in ner_dict:
        eval_dict["B"+mid+item] = [0]*EVAL_ITEM_LEN
        eval_dict["I"+mid + item] = [0] * EVAL_ITEM_LEN
        eval_dict[item] = [0] * EVAL_ITEM_LEN
    eval_dict[UNK_TAG]=[0]*EVAL_ITEM_LEN
    for index, item in tqdm(true_csv.iterrows()):
        true_line = item[TAG_COL].split()
        pred_line = pred_csv.loc[index][TAG_COL].split()

        #calculate preds
        true_word = item[WORD_COL].split()
        pred_word = pred_csv.loc[index][WORD_COL].split()
        true_tag = get_ner_word(true_word,true_line)
        pred_tag = get_ner_word(pred_word,pred_line)
        all+=len(true_tag)
        for item in pred_tag:
            if item in true_tag:
                preds+=1

        # print(true_line)
        # print(pred_line)
        for i_index in range(len(true_line)):
            # print(true_line[i_index],pred_line[i_index])
            assert true_line[i_index] in eval_dict.keys() and pred_line[i_index] in eval_dict.keys() \
                , 'the pred tag is not in dict!'
            if true_line[i_index] == pred_line[i_index]:
                eval_dict[true_line[i_index]][TP] += 1
                if len(true_line[i_index])>1:
                    eval_dict[true_line[i_index][2:]][TP] += 1
            else:
                eval_dict[true_line[i_index]][FN] += 1
                eval_dict[pred_line[i_index]][FP] += 1
                if len(true_line[i_index])>1:
                    eval_dict[true_line[i_index][2:]][FN] += 1
                if len(pred_line[i_index])>1:
                    eval_dict[pred_line[i_index][2:]][FP] += 1

    for item in eval_dict:
        eval_dict[item][RECALL] = eval_dict[item][TP]/(eval_dict[item][TP]+eval_dict[item][FP]) \
            if eval_dict[item][TP]>0 else 0.0
        eval_dict[item][PRECISION] = eval_dict[item][TP] / (eval_dict[item][TP] + eval_dict[item][FN]) \
            if eval_dict[item][TP]>0 else 0.0
        eval_dict[item][F1_SCORE]=2 * eval_dict[item][PRECISION]*eval_dict[item][RECALL]/(eval_dict[item][PRECISION]+eval_dict[item][RECALL])\
            if eval_dict[item][PRECISION]+eval_dict[item][RECALL] > 0 else 0.0

    pprint(eval_dict)

    # print("all tags:{}, pred tags:{}, pro:{}".format(all,preds,float(preds/all)))
    return eval_dict,[all,preds,float(preds/all)]


def eval():
    # true_csv = pd.read_csv('../CRF/preds/weibo_test_gold.csv')
    # pred_csv = pd.read_csv('../CRF/preds/weibo_test_pred.csv')
    # ner_dict = load_pkl('../BiLSTM_CRF/models/weibo_ner_params.pkl')['train_tagset']
    # print(ner_dict)
    # res = get_pred_main(true_csv, pred_csv, ner_dict, mid='-')





    pass

if __name__ == '__main__':
    true_csv = pd.read_csv('../CRF/preds/clue_gold.csv')
    pred_csv = pd.read_csv('../CRF/preds/clue_pred.csv')
    ner_dict = load_pkl('../BiLSTM_CRF/models/clue_ner_params.pkl')['train_tagset']
    print(ner_dict)
    res = get_pred_main(true_csv, pred_csv, ner_dict, mid='_')


    pass