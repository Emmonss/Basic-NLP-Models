import os, sys
import tensorflow as tf
import pandas as pd
import numpy as np

sys.path.append('../')
sys.path.append('../../')

from BasicLayerModels.RNNs.layers.ConditionalRandomField import CRF
from Segment.DataProcess.data import get_words_label_data
from tqdm import tqdm
from pprint import pprint
from tensorflow.keras.models import load_model
from Segment.Utils.basic_utils import *
from Segment.DataProcess.data import WORD_COL, TAG_COL

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

PRED_RES_PATH_ROOT = './preds'


def get_model(model_path):
    model = load_model(model_path, custom_objects=None)
    model.summary()
    return model


def get_batch_trans(batch_data, batch_gold, batch_pred, IndexWordDict, IndexTagDict):
    words = []
    gold_tags = []
    pred_tags = []
    for index in range(len(batch_data)):
        word = index2word(batch_data[index], IndexWordDict)
        words.append(word)
        gold_tags.append(index2tag(batch_gold[index], IndexTagDict, len(word.split())))
        pred_tags.append(index2tag(batch_pred[index], IndexTagDict, len(word.split())))
    return words, gold_tags, pred_tags


def predict_main(model_path, param_path, val_path, save_path_head, batch_size=32):
    words_all = []
    gold_tags_all = []
    pred_tags_all = []

    train_param = load_pkl(param_path)
    pprint(train_param.keys())
    model = get_model(model_path=model_path)
    IndexWordDict = reverse_dict(train_param['train_wordIndexDict'])
    IndexTagDict = reverse_dict(train_param['train_tagIndexDict'])
    _, _, _, _, _, _, val_X, val_y = get_words_label_data(val_path,
                                                          super_wordIndexDict=train_param['train_wordIndexDict'],
                                                          super_tagIndexDict=train_param['train_tagIndexDict'],
                                                          super_max_len=train_param['train_maxLen'],
                                                          val_flag=True)
    val_size = np.shape(val_X)[0]
    print("val_size:{}".format(val_size))
    for i in tqdm(range(int(val_size / batch_size))):
        batch_data = val_X[i * batch_size:min(val_size, (i + 1) * batch_size)]
        batch_gold = val_y[i * batch_size:min(val_size, (i + 1) * batch_size)]
        batch_pred = model.predict(batch_data)

        words, gold_tags, pred_tags = get_batch_trans(batch_data, batch_gold, batch_pred, IndexWordDict, IndexTagDict)
        words_all.extend(words)
        gold_tags_all.extend(gold_tags)
        pred_tags_all.extend(pred_tags)

    dict_gold = {
        WORD_COL: words_all,
        TAG_COL: gold_tags_all
    }

    dict_pred = {
        WORD_COL: words_all,
        TAG_COL: pred_tags_all
    }

    if not os.path.exists(PRED_RES_PATH_ROOT):
        os.mkdir(PRED_RES_PATH_ROOT)

    pd.DataFrame(dict_gold). \
        to_csv(os.path.join(PRED_RES_PATH_ROOT, '{}_gold.csv'.format(save_path_head)), index=False)
    pd.DataFrame(dict_pred). \
        to_csv(os.path.join(PRED_RES_PATH_ROOT, '{}_pred.csv'.format(save_path_head)), index=False)


################################################################################################
# train data mode
################################################################################################
def get_msr_data_predict():
    model_path = './models/msr_bilstm_softmax.h5'
    param_path = './models/msr_params.pkl'
    val_path = '../datas/ProcessData/msr_test.csv'
    predict_main(model_path=model_path, param_path=param_path, val_path=val_path,
                 save_path_head='msr_test')


def get_pku_data_predict():
    model_path = './models/pku_bilstm_softmax.h5'
    param_path = './models/pku_params.pkl'
    val_path = '../datas/ProcessData/pku_data.csv'
    predict_main(model_path=model_path, param_path=param_path, val_path=val_path,
                 save_path_head='pku_test')


################################################################################################

if __name__ == '__main__':
    # get_pku_data_predict()

    get_msr_data_predict()
    pass