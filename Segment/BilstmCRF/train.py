import sys,os
sys.path.append('../../')

from Basic_Layer_Models.RNNs.models.BiLSTM_CRF import BiLSTM_CRF
from Segment.DataProcess.data import get_words_label_data

'''
                 vocab_size,
                 embeddings,
                 hidden_units,
                 tag_num,
                 seg_max_len,
                 lr,
'''
def load_model(vocab_size,
               embeddings,
               hidden_units,
               tag_num,
               seg_max_len):
    model = BiLSTM_CRF(vocab_size,
                 embeddings,
                 hidden_units,
                 tag_num,
                 seg_max_len,
                 lr=0.01,)

    return model


def train():
    train_path = './ProcessData/msr_train.csv'
    val_path = './ProcessData/msr_test.csv'

    train_wordIndexDict,train_vocabSize,train_maxLen,train_sequenceLengths,\
    train_tagSum,train_tagIndexDict,train_X,train_y = get_words_label_data(train_path)

    val_wordIndexDict, val_vocabSize, val_maxLen, val_sequenceLengths, \
    val_tagSum, val_tagIndexDict, val_X, val_y = get_words_label_data(val_path)

    print("="*30+"train data"+'='*30)
    print("X shape:{}".format(train_X.shape))
    print("y shape:{}".format(train_y.shape))

    print("=" * 30 + "val data" + '=' * 30)
    print("X shape:{}".format(val_X.shape))
    print("y shape:{}".format(val_y.shape))

    model = load_model(vocab_size=train_vocabSize,
               embeddings=300,
               hidden_units=512,
               tag_num=train_tagSum,
               seg_max_len =train_maxLen)

    model.fit(train_X,train_y,valid_data=(val_X,val_y))


if __name__ == '__main__':
    train()