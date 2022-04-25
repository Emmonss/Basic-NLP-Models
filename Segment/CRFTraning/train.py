import pycrfsuite,os
import pandas as pd

from tqdm import tqdm
from Segment.CRFTraning.mode import *
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
def get_msr_data_train_save():
    param = {
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 200,
        'feature.possible_transitions': True,
        'feature.minfreq': 3
    }
    train(filename='../datas/ProcessData/msr_train.csv', params=param,
          mode_path='./models', model_file='msr_crf.crfsuite')
    pass

def get_pku_data_train_save():
    param = {
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 200,
        'feature.possible_transitions': True,
        'feature.minfreq': 3
    }
    train(filename='../datas/ProcessData/pku_data.csv', params=param,
          mode_path='./models', model_file='pku_crf.crfsuite')
    pass
################################################################################################
if __name__ == '__main__':
    get_pku_data_train_save()
    pass