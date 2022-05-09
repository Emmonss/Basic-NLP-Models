import sys,os
import tensorflow as tf
sys.path.append('../../')
sys.path.append('../')
from Basic_Layer_Models.RNNs.models.BiLSTM_CRF import BiLSTM_CRF
from Segment.DataProcess.data import get_words_label_data
from Segment.Utils.basic_utils import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
    except RuntimeError as e:
        print(e)

def load_model(vocab_size,
               embeddings,
               hidden_units,
               tag_num,
               seg_max_len,
               lr):
    model = BiLSTM_CRF(vocab_size,
                 embeddings,
                 hidden_units,
                 tag_num,
                 seg_max_len,
                 lr=lr,)

    return model


def train(train_path,val_path=None,mode='val',
          embeddings=300,hidden_units=512,batch_size=32,lr=0.01,epoch=20,val_split=0.1,
          save_flag=False,model_path=None,model_name=None,param_name=None):

    train_wordIndexDict,train_vocabSize,train_maxLen,train_sequenceLengths,\
    train_tagSum,train_tagIndexDict,train_X,train_y = get_words_label_data(train_path)

    print("="*30+"train data"+'='*30)
    print("X shape:{}".format(train_X.shape))
    print("y shape:{}".format(train_y.shape))

    model = load_model(vocab_size=train_vocabSize,
                       embeddings=embeddings,
                       hidden_units=hidden_units,
                       tag_num=train_tagSum,
                       seg_max_len=train_maxLen,
                       lr=lr)

    if mode=='val':
        val_wordIndexDict, val_vocabSize, val_maxLen, val_sequenceLengths, \
        val_tagSum, val_tagIndexDict, val_X, val_y = get_words_label_data(val_path,
                                                                          super_max_len=train_maxLen,
                                                                          super_tagIndexDict=train_tagIndexDict,
                                                                          super_wordIndexDict=train_wordIndexDict,
                                                                          val_flag=True)
        print("=" * 30 + "val data" + '=' * 30)
        print("X shape:{}".format(val_X.shape))
        print("y shape:{}".format(val_y.shape))
        model.fit_val(train_X,train_y,valid_data=(val_X,val_y),batch_size=batch_size,epoch=epoch)
        #msr的验证集太大了，太慢了，可以考虑还是split去验证
        # model.fit_val(train_X,train_y,valid_data=val_split,batch_size=batch_size,epoch=epoch)
    elif mode=='train':
        model.fit_train(train_X, train_y,val_split=val_split, batch_size=batch_size, epoch=epoch)

    if not (save_flag==False or model_name==None or model_path==None or param_name==None):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model.save(model_path=model_path,model_name=model_name)

        train_param_dict = {
            'train_wordIndexDict':train_wordIndexDict,
            'train_vocabSize':train_vocabSize,
            'train_maxLen':train_maxLen,
            'train_tagSum':train_tagSum,
            'train_tagIndexDict':train_tagIndexDict
        }
        save_pkl(os.path.join(model_path,'{}.pkl'.format(param_name)),train_param_dict)


################################################################################################
# train data mode
################################################################################################
def get_msr_data_train_save():
    train_path = '../datas/ProcessData/msr_train.csv'
    val_path = '../datas/ProcessData/msr_test.csv'
    train(train_path, val_path,mode='val',
          save_flag=True,model_path='./models',model_name='msr_bilstm_crf',param_name='msr_params',
          epoch=50,lr=0.0001)

def get_pku_data_train_save():
    train_path = '../datas/ProcessData/pku_data.csv'
    train(train_path, mode='train',
          save_flag=True, model_path='./models', model_name='pku_bilstm_crf',param_name='pku_params',
          epoch=50,lr=0.0001)

################################################################################################

if __name__ == '__main__':
    # get_pku_data_train_save()

    get_msr_data_train_save()
    pass