
import numpy as np
import pandas as pd
import re,sys
sys.path.append('../../')

from Segment.DataProcess.data_utils import back_trans_sentence
from Segment.DataProcess.data import WORD_COL,TAG_COL
from pprint import pprint
from tqdm import tqdm


def get_precision(y_gold,y_true):
    return float(len(y_gold) / len(y_true))

def get_recall(y_gold,y_pred):
    return float(len(y_gold)/len(y_pred))

def get_fl(precision,recall):
    return float(2*precision*recall/(precision+recall))

def get_correct(y_true_1,y_pred_1):
    res = []
    for word in y_pred_1:
        if word in y_true_1:
            res.append(word)
    return res

def get_word_accuracy_single(list1,list2):
    assert len(list1) == len(list2), 'the length is not equal'
    acc = 0
    for index in range(len(list1)):
        if list1[index]==list2[index]:
           acc+=1
    return acc,len(list2)

def to_region(segmentation):
    region = []
    start = 0
    seg = ''
    for word in segmentation:
        seg+='{} '.format(word)

    for word in re.compile("\\s+").split(seg.strip()):
        end = start + len(word)
        region.append((start, end))
        start = end
    return region

def get_oov_iv(y_true,y_pred,word_dict):
    assert len(y_pred) == len(y_true), 'the length is not equal'

    oov,oov_r,iv,iv_r=0,0,0,0
    for index in tqdm(range(len(y_true))):
        y_true_region = to_region(y_true[index])
        y_pred_region = to_region(y_pred[index])

        text = ""
        for item in y_true[index]:
            text+=item

        for (start, end) in y_true_region:
            word = text[start: end]
            if word in word_dict:
                iv += 1
            else:
                oov += 1


        for (start, end) in set(y_true_region) & set(y_pred_region):
            word = text[start: end]
            if word in word_dict:
                iv_r += 1
            else:
                oov_r += 1

    return float(iv_r/iv) if iv>0 else 0.0,float(oov_r/oov) if oov>0 else 0.0

def get_word_accuracy(y_true,y_pred):
    assert len(y_pred) == len(y_true), 'the length is not equal'
    acc_count,all_count=0,0
    for index in range(len(y_true)):
        acc_count_1,all_count_1 =get_word_accuracy_single(y_true[index],y_pred[index])
        acc_count+=acc_count_1
        all_count+=all_count_1
    return acc_count,all_count,float(acc_count/all_count)

def get_segment_pred(y_true,y_pred):
    assert len(y_pred)==len(y_true),'the length is not equal'
    true_all_count = []
    pred_all_count = []
    correct_all_count =[]
    for index in range(len(y_true)):
        correct_1 = get_correct(y_pred[index],y_true[index])
        true_all_count+=y_true[index]
        pred_all_count+=y_pred[index]
        correct_all_count+=correct_1
    precision = get_precision(correct_all_count,true_all_count)
    recall = get_precision(correct_all_count, pred_all_count)
    f1_score = get_fl(precision,recall)

    return precision,recall,f1_score

def get_pred_main(true_csv,pred_csv):
    assert len(true_csv)==len(pred_csv),'length is not equal'
    gold_dict = []
    gold_list = []
    gold_tag = []
    for index,item in tqdm(true_csv.iterrows()):
        sents = back_trans_sentence(sentence=item[WORD_COL],tags=item[TAG_COL])
        gold_list.append(sents)
        gold_tag.append(item[TAG_COL].strip().split())
        gold_dict.extend(sents)
    gold_dict = list(set(gold_dict))

    pred_list = []
    pred_tag = []
    for index, item in tqdm(pred_csv.iterrows()):
        sents = back_trans_sentence(sentence=item[WORD_COL], tags=item[TAG_COL])
        pred_list.append(sents)
        pred_tag.append(item[TAG_COL].strip().split())

    word_acc_count, word_all_count, word_acc = get_word_accuracy(y_true=gold_tag,y_pred=pred_tag)
    print('acc done')

    precision, recall, f1_score = get_segment_pred(y_true=gold_list,y_pred=pred_list)
    print("f1 done")

    iv,oov = get_oov_iv(y_true=gold_list,y_pred=pred_list,word_dict=gold_dict)
    print('iv oov done')
    # iv=0.0
    # oov=0.0

    res_dict = {
        'word_acc':word_acc,
        'presicion':precision,
        'recall':recall,
        "f1_score":f1_score,
        "IV":iv,
        'OOV':oov
    }
    return res_dict


if __name__ == '__main__':
    true_csv = pd.read_csv('./data/test_gold.csv')
    pred_csv = pd.read_csv('./data/test_pred.csv')
    res = get_pred_main(true_csv,pred_csv)
    pprint(res)
    pass