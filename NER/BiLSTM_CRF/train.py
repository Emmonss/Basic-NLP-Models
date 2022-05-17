import sys,os
import tensorflow as tf
sys.path.append('../../')
sys.path.append('../')
from Basic_Layer_Models.RNNs.models.BiLSTM_CRF import BiLSTM_CRF
from NER.DataProcess.data import get_words_label_data

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