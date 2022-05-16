import pycrfsuite,os,sys
sys.path.append('../')
sys.path.append('../../')
from NER.CRF.data import *
from NER.CRF.model import *
from NER.DataProcess.data import WORD_COL,TAG_COL
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
def get_msra_predict_save():
    model_path = './models/msra.crfsuite'
    val_path = '../data/Proessdata/msra_test.csv'
    prediction(val_path=val_path,
               model_path=model_path,
               pred_name_head='msra')
    pass

def get_boston_predict_save():
    model_path = './models/boston.crfsuite'
    val_path = '../data/Proessdata/boston.csv'
    prediction(val_path=val_path,
               model_path=model_path,
               pred_name_head='boston')
    pass

def get_clue_predict_save():
    model_path = './models/clue.crfsuite'
    val_path = '../data/Proessdata/clue_dev.csv'
    prediction(val_path=val_path,
               model_path=model_path,
               pred_name_head='clue')
    pass

def get_weibo_predict_save():
    model_path = './models/weibo.crfsuite'
    val_path = '../data/Proessdata/weibo_val.csv'
    prediction(val_path=val_path,
               model_path=model_path,
               pred_name_head='weibo_val')

    val_path = '../data/Proessdata/weibo_test.csv'
    prediction(val_path=val_path,
               model_path=model_path,
               pred_name_head='weibo_test')
    pass

################################################################################################


if __name__ == '__main__':
    get_weibo_predict_save()
    pass