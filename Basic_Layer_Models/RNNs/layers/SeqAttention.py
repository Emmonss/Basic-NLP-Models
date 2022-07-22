import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input
from Basic_Layer_Models.RNNs.layers.SeqEncoder import EncoderSingleLSTM
import numpy as np


class BahdanauAttention(Model):
    def __init__(self,units,method='general'):
        super(BahdanauAttention, self).__init__()
        assert method in ['dot','general','concat'],"attention method error"
        self.w1 = Dense(units)
        self.w2 = Dense(units)
        self.v = Dense(1)
        self.method = method

    def call(self,decoder_hidden, encoder_outputs):
        '''
        :param decoder_hidden: shape:[None, hidden_units]
        :param encoder_outputs: shape:[None, seq_len, hidden_units]
        :return: shape:[batch_size,encoder_max_len]
        '''
        if self.method == "dot":
            return self.dot_score(decoder_hidden, encoder_outputs)

        elif self.method == "general":
            return self.general_socre(decoder_hidden, encoder_outputs)

        elif self.method == "concat":
            return self.concat_socre(decoder_hidden, encoder_outputs)

    def dot_score(self,decoder_hidden, encoder_outputs):
        '''
        :param decoder_hidden: shape:[None, hidden_units]
        :param encoder_outputs: shape:[None, seq_len, hidden_units]
        :return:
        '''
        print("decoder_hidden shape:{}".format(np.shape(decoder_hidden)))
        print("encoder_outputs shape:{}".format(np.shape(encoder_outputs)))
        #[None, hidden_units]
        decoder_hidden_1 = tf.expand_dims(decoder_hidden,-1)
        print("decoder_hidden_1 shape:{}".format(np.shape(decoder_hidden_1)))

        y1 = tf.matmul(encoder_outputs, decoder_hidden_1)
        print("y1 shape:{}".format(np.shape(y1)))

        y2 = tf.squeeze(y1,axis=-1)
        print("y2 shape:{}".format(np.shape(y2)))
        return None
        pass

    def general_socre(self, decoder_hidden, encoder_outputs):
        return None

    def concat_socre(self, decoder_hidden, encoder_outputs):
        return None

if __name__ == '__main__':
    # inputs = Input(shape=(50,), name='inputs')
    # encoder_outputs, state_h, state_c = EncoderSingleLSTM(vocab_size=1000,
    #                                             embeddin_dim=512,
    #                                             hidden_units=300)(inputs)
    encoder_outputs,state_h = tf.random.normal([32,5,300]),\
                              tf.random.normal([32,300])
    attention_weights = BahdanauAttention(units=300,method='dot')(state_h,encoder_outputs)
    # print(np.shape(context_vector))
    #
    # model = Model(inputs, (context_vector,attention_weights))
    # t.build()
    # model.summary()












