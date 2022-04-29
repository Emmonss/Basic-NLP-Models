import os,sys
sys.path.append('../')
sys.path.append('../../')

import pandas as pd
from pprint import pprint
from Segment.Utils.SegmentUtils import get_pred_main


def eval_main(true_csv_path,pred_csv_path):
    true_csv = pd.read_csv(true_csv_path)
    pred_csv = pd.read_csv(pred_csv_path)
    res = get_pred_main(true_csv, pred_csv)
    pprint(res)


################################################################################################
# train data mode
################################################################################################
def get_msr_data_eval():
    true_csv_path = './preds/msr_gold.csv'
    pred_csv_path = './preds/msr_pred.csv'
    eval_main(true_csv_path,pred_csv_path)

def get_pku_data_eval():
    true_csv_path = './preds/pku_gold.csv'
    pred_csv_path = './preds/pku_pred.csv'
    eval_main(true_csv_path, pred_csv_path)

def get_pku_msr_data_eval():
    true_csv_path = './preds/pku_train_msr_test_gold.csv'
    pred_csv_path = './preds/pku_train_msr_test_pred.csv'
    eval_main(true_csv_path, pred_csv_path)

def get_msr_pku_data_eval():
    true_csv_path = './preds/msr_train_pku_test_gold.csv'
    pred_csv_path = './preds/msr_train_pku_test_pred.csv'
    eval_main(true_csv_path, pred_csv_path)

################################################################################################


if __name__ == '__main__':
    # get_msr_data_eval()
    # get_pku_data_eval()
    # get_pku_msr_data_eval()
    # get_msr_pku_data_eval()
    pass