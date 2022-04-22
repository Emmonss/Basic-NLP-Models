import os,sys
sys.path.append('../')

import pandas as pd
import tensorflow as tf

from pprint import pprint
from Segment.Utils.SegmentUtils import get_pred_main

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
    except RuntimeError as e:
        print(e)


def eval_main(true_csv_path,pred_csv_path):
    true_csv = pd.read_csv(true_csv_path)
    pred_csv = pd.read_csv(pred_csv_path)
    res = get_pred_main(true_csv, pred_csv)
    pprint(res)


################################################################################################
# train data mode
################################################################################################
def get_msr_data_eval():
    true_csv_path = './preds/msr_test_gold.csv'
    pred_csv_path = './preds/msr_test_pred.csv'
    eval_main(true_csv_path,pred_csv_path)

def get_pku_data_eval():
    pass

################################################################################################


if __name__ == '__main__':
    get_msr_data_eval()