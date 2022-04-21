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

if __name__ == '__main__':
    true_csv = pd.read_csv('./preds/msr_test_gold.csv')
    pred_csv = pd.read_csv('./preds/msr_test_pred.csv')
    res = get_pred_main(true_csv,pred_csv)
    pprint(res)