import pycrfsuite,os,sys
sys.path.append('../')
sys.path.append('../../')
from Segment.CRFTraning.data import *
from Segment.CRFTraning.model import *
from Segment.DataProcess.data import WORD_COL,TAG_COL
from tqdm import tqdm

PRED_RES_PATH_ROOT = './preds'

def prediction(val_path,model_path,pred_name_head):
    test_set = LoadData(val_path)

    tagger = pycrfsuite.Tagger()
    tagger.open(model_path)

    words = []
    preds = []
    golds = []
    for item in tqdm(test_set):
        words.append(' '.join(sent2tokens(item)).strip())
        preds.append(' '.join(tagger.tag(sent2features(item))).strip())
        golds.append(' '.join(sent2labels(item)).strip())

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
        to_csv(os.path.join(PRED_RES_PATH_ROOT, '{}_gold.csv'.format(pred_name_head)), index=False)
    pd.DataFrame(dict_pred). \
        to_csv(os.path.join(PRED_RES_PATH_ROOT, '{}_pred.csv'.format(pred_name_head)), index=False)

################################################################################################
# train data mode
################################################################################################
def get_msr_data_predict_save():
    model_path = './models/msr_crf.crfsuite'
    val_path = '../datas/ProcessData/msr_test.csv'
    prediction(val_path=val_path,
               model_path=model_path,
               pred_name_head='msr')
    pass

def get_pku_data_predict_save():
    model_path = './models/pku_crf.crfsuite'
    val_path = '../datas/ProcessData/pku_data.csv'
    prediction(val_path=val_path,
               model_path=model_path,
               pred_name_head='pku')
    pass

def get_pku_msr_predict_save():
    model_path = './models/pku_crf.crfsuite'
    val_path = '../datas/ProcessData/msr_test.csv'
    prediction(val_path=val_path,
               model_path=model_path,
               pred_name_head='pku_train_msr_test')

def get_msr_pku_predict_save():
    model_path = './models/msr_crf.crfsuite'
    val_path = '../datas/ProcessData/pku_data.csv'
    prediction(val_path=val_path,
               model_path=model_path,
               pred_name_head='msr_train_pku_test')
################################################################################################


if __name__ == '__main__':
    # get_msr_data_predict_save()
    # get_pku_data_predict_save()
    get_pku_msr_predict_save()
    get_msr_pku_predict_save()
    pass