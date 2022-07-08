import os
from Basic_Layer_Models.RNNs.models.NLU_Basic import NLUModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Embedding,Bidirectional,LSTM
import tensorflow as tf
from Basic_Layer_Models.RNNs.layers.ConditionalRandomField import CRF

class BiLSTM_CRF(NLUModel):
    def __init__(self,
                 vocab_size,
                 embeddings,
                 hidden_units,
                 tag_num,
                 seg_max_len,
                 lr=0.001,
                 opt='Adam'):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.embeddings = embeddings
        self.hiden_units = hidden_units
        self.tag_sum = tag_num
        self.seg_max_len = seg_max_len

        self.opt = self.get_opts(opt,lr)

        self.build_model()
        self.compile_model()

    def build_model(self):
        inputs = Input(shape=(self.seg_max_len,),name='inputs')
        embed = Embedding(self.vocab_size, self.embeddings,name='embed')(inputs)
        bi_lstm1 = Bidirectional(LSTM(self.hiden_units, return_sequences=True, activation="tanh"),
                                merge_mode='sum',name='bilistm_1')(embed)
        bi_lstm2 = Bidirectional(LSTM(self.tag_sum, return_sequences=True, activation="softmax"),
                                merge_mode='sum',name='bilistm_2')(bi_lstm1)
        self.crf = CRF(self.tag_sum, name='crf_layer')
        output =self.crf(bi_lstm2)

        self.model = Model(inputs, output)

    def compile_model(self):
        self.model.compile(optimizer=self.opt,
                           loss={'crf_layer':self.crf.get_loss},
                           loss_weights={'crf_layer':1.0},
                           metrics={'crf_layer':'acc'})
        self.model.summary()



if __name__ == '__main__':
    BiLSTM_CRF(vocab_size=5000,
           embeddings=300,
           hidden_units=300,
           seg_max_len=50,
           tag_num=5,
           lr=0.01
           )