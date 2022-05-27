import os,sys
sys.path.append('../')
sys.path.append('../../')

import pandas as pd
from pprint import pprint
from NER.Utils.NER_Utils import get_pred_main
from NER.Utils.basic_utils import load_pkl

################################################################################################
# train data mode
################################################################################################
def get_prd_boston():
    true_csv = pd.read_csv('./preds/boston_gold.csv')
    pred_csv = pd.read_csv('./preds/boston_pred.csv')
    ner_dict = load_pkl('../BiLSTM_CRF/models/boston_ner_params.pkl')['train_tagset']
    print(ner_dict)
    eval_score,tag_pred = get_pred_main(true_csv, pred_csv,ner_dict,mid='_')
    pprint(tag_pred)
    print("all tags:{}, pred tags:{}, pro:{}".format(tag_pred[0], tag_pred[1], tag_pred[2]))
    return eval_score,tag_pred

def get_prd_clue():
    true_csv = pd.read_csv('../CRF/preds/clue_gold.csv')
    pred_csv = pd.read_csv('../CRF/preds/clue_pred.csv')
    ner_dict = load_pkl('../BiLSTM_CRF/models/clue_ner_params.pkl')['train_tagset']
    print(ner_dict)
    eval_score,tag_pred = get_pred_main(true_csv, pred_csv, ner_dict, mid='_')
    pprint(eval_score)
    print("all tags:{}, pred tags:{}, pro:{}".format(tag_pred[0], tag_pred[1], tag_pred[2]))
    return eval_score,tag_pred

def get_prd_msra():
    true_csv = pd.read_csv('../CRF/preds/msra_gold.csv')
    pred_csv = pd.read_csv('../CRF/preds/msra_pred.csv')
    ner_dict = load_pkl('../BiLSTM_CRF/models/msra_ner_params.pkl')['train_tagset']
    print(ner_dict)
    eval_score, tag_pred = get_pred_main(true_csv, pred_csv, ner_dict, mid='_')
    pprint(eval_score)
    print("all tags:{}, pred tags:{}, pro:{}".format(tag_pred[0], tag_pred[1], tag_pred[2]))
    return eval_score, tag_pred

def get_prd_weibo_test():
    true_csv = pd.read_csv('../CRF/preds/weibo_test_gold.csv')
    pred_csv = pd.read_csv('../CRF/preds/weibo_test_pred.csv')
    ner_dict = load_pkl('../BiLSTM_CRF/models/weibo_ner_params.pkl')['train_tagset']
    print(ner_dict)
    eval_score, tag_pred = get_pred_main(true_csv, pred_csv, ner_dict, mid='-')
    pprint(eval_score)
    print("all tags:{}, pred tags:{}, pro:{}".format(tag_pred[0], tag_pred[1], tag_pred[2]))
    return eval_score, tag_pred

def get_prd_weibo_val():
    true_csv = pd.read_csv('../CRF/preds/weibo_val_gold.csv')
    pred_csv = pd.read_csv('../CRF/preds/weibo_val_pred.csv')
    ner_dict = load_pkl('../BiLSTM_CRF/models/weibo_ner_params.pkl')['train_tagset']
    print(ner_dict)
    eval_score, tag_pred = get_pred_main(true_csv, pred_csv, ner_dict, mid='-')
    pprint(eval_score)
    print("all tags:{}, pred tags:{}, pro:{}".format(tag_pred[0], tag_pred[1], tag_pred[2]))
    return eval_score, tag_pred

################################################################################################

if __name__ == '__main__':
    get_prd_clue()
    pass