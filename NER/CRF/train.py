import pycrfsuite,os
import pandas as pd

from tqdm import tqdm
from Segment.CRFTraning.model import *
from Segment.CRFTraning.data import *

def train(filename,params,model_file=None,mode_path =None):
    train_set = LoadData(filename)
    trainer = pycrfsuite.Trainer(verbose=False)

    for item in tqdm(train_set):
        trainer.append(sent2features(item), sent2labels(item))
    trainer.set_params(params)
    print("traning mode.........")
    trainer.train(os.path.join(mode_path,model_file))
    print("done!")
    pass


################################################################################################
# train data mode
################################################################################################
def get_pd2014_train_save():
    param = {
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 200,
        'feature.possible_transitions': True,
        'feature.minfreq': 3
    }
    train(filename='../data/Proessdata/pd2014.csv', params=param,
          mode_path='./models', model_file='pd2014.crfsuite')

def get_msra_train_save():
    param = {
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 200,
        'feature.possible_transitions': True,
        'feature.minfreq': 3
    }
    train(filename='../data/Proessdata/msra_train.csv', params=param,
          mode_path='./models', model_file='msra.crfsuite')
    pass

def get_boston_train_save():
    param = {
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 200,
        'feature.possible_transitions': True,
        'feature.minfreq': 3
    }
    train(filename='../data/Proessdata/boston.csv', params=param,
          mode_path='./models', model_file='boston.crfsuite')
    pass

def get_weibo_train_save():
    param = {
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 200,
        'feature.possible_transitions': True,
        'feature.minfreq': 3
    }
    train(filename='../data/Proessdata/weibo_train.csv', params=param,
          mode_path='./models', model_file='weibo.crfsuite')
    pass
################################################################################################
if __name__ == '__main__':
    get_weibo_train_save()
    pass