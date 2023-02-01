from EvalUtils.logUtils import Logger
from SimSentence.BertOnly.sentsToken import read_corpus
from BasicLayerModels.Transformer.models.SimSentBertModel import BertModelForSimsent
from SimSentence.BertOnly.tokenizer import SimBertTokenizer
from SimSentence.BertOnly import trainConfig as config
import tensorflow as tf
import numpy as np


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
    except RuntimeError as e:
        print(e)

logger = Logger("Main").get_logger()


def load_model():
    model = BertModelForSimsent(config)
    return model

def train(train_data_path,val_data_path,tokenizer):
    logger.info("loading and trans data ")
    train_data = read_corpus(train_data_path)
    train_token_ids, train_seg_ids, train_tags = tokenizer.SimSent2BertIndex(train_data)
    logger.info("train token_ids:{},tags;{}".format(np.shape(train_token_ids),np.shape(train_tags)))


    val_data = read_corpus(val_data_path)
    val_token_ids, val_seg_ids, val_tags = tokenizer.SimSent2BertIndex(val_data)
    logger.info("train token_ids:{},tags;{}".format(np.shape(val_token_ids), np.shape(val_tags)))

    train_X = [train_token_ids, train_seg_ids]
    train_Y = (train_tags)
    val_X = [val_token_ids, val_seg_ids]
    val_Y = (val_tags)

    logger.info("load model")
    model = load_model()

    logger.info("start training")
    model.fit(train_X,train_Y,
                  valid_data=(val_X,val_Y),
                  epochs=config.epoch,
                  batch_size=config.batch_size)

#######################################################
def train_bq_corpus(tokenizer):

    logger.info("train_bq_corpus simsententce")
    train_data_path = '../datas/bq_corpus/train.tsv'
    val_data_path = '../datas/bq_corpus/dev.tsv'
    train(train_data_path,val_data_path,tokenizer)

if __name__ == '__main__':
    pass
    tokenizer = SimBertTokenizer(config.vocab_path,
                                 maxlen=config.max_len,
                                 do_lower_case=config.do_lower_case)
    train_bq_corpus(tokenizer)

