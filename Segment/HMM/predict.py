import sys,os
sys.path.append('../')

import pandas as pd
from Segment.HMM.Vertebi import Vertebi
from Segment.DataProcess.data import WORD_COL,TAG_COL
from tqdm import tqdm

PRED_RES_PATH_ROOT ='./preds'

def predict_main(data_path,pred_name,matrix_path,mode='load'):
    hmm_vertebi = Vertebi(mode=mode,model_path=matrix_path)
    assert hmm_vertebi.hmm.matrix_flag,"the model has some wrong"
    data = pd.read_csv(data_path)

    words = []
    preds = []
    golds = []
    for index,item in tqdm(data.iterrows()):
        ftags = ''
        sentence = item[WORD_COL].replace(" ","").strip()
        tags = hmm_vertebi.get_segment_tags(sentence)
        for t in tags:
            ftags += t+' '
        ftags= ftags.strip()
        words.append(item[WORD_COL])
        preds.append(ftags)
        golds.append(item[TAG_COL])

    dict_gold = {
        WORD_COL: words,
        TAG_COL: golds
    }

    dict_pred = {
        WORD_COL: words,
        TAG_COL: preds
    }

    if not os.path.exists(PRED_RES_PATH_ROOT):
        os.mkdir(PRED_RES_PATH_ROOT)

    pd.DataFrame(dict_gold). \
        to_csv(os.path.join(PRED_RES_PATH_ROOT, '{}_gold.csv'.format(pred_name)), index=False)
    pd.DataFrame(dict_pred). \
        to_csv(os.path.join(PRED_RES_PATH_ROOT, '{}_pred.csv'.format(pred_name)), index=False)


################################################################################################
# train data mode
################################################################################################
def get_msr_data_train_save():
    val_path = '../datas/ProcessData/msr_test.csv'
    pred_name = "msr"
    matrix_path = './matrixs/msr_hmm.pkl'
    predict_main(data_path=val_path,pred_name=pred_name,matrix_path=matrix_path)

def get_pku_data_train_save():
    val_path = '../datas/ProcessData/pku_data.csv'
    pred_name = "pku"
    matrix_path = './matrixs/pku_hmm.pkl'
    predict_main(data_path=val_path, pred_name=pred_name, matrix_path=matrix_path)

def get_pku_msr_train_save():
    val_path = '../datas/ProcessData/msr_test.csv'
    pred_name = "pku_train_msr_test"
    matrix_path = './matrixs/pku_hmm.pkl'
    predict_main(data_path=val_path, pred_name=pred_name, matrix_path=matrix_path)

def get_msr_pku_train_save():
    val_path = '../datas/ProcessData/pku_data.csv'
    pred_name = "msr_train_pku_test"
    matrix_path = './matrixs/msr_hmm.pkl'
    predict_main(data_path=val_path, pred_name=pred_name, matrix_path=matrix_path)

################################################################################################


if __name__ == '__main__':
    get_msr_data_train_save()
    get_pku_data_train_save()
    get_pku_msr_train_save()
    get_msr_pku_train_save()
    pass
