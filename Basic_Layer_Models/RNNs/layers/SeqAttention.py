import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input
import numpy as np


class Attention(Model):
    def __init__(self,units,method='general'):
        super(Attention, self).__init__()
        assert method in ['dot','general','concat','concat_diverse'],"attention method error"

        self.method = method
        if self.method=='general':
            self.w = Dense(units)
        elif self.method =='concat':
            self.w = Dense(units)
            self.v = Dense(1)
        elif self.method =='concat_diverse':
            self.w1 = Dense(units)
            self.w2 = Dense(units)
            self.v = Dense(1)

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
        elif self.method == "concat_diverse":
            return self.concat_diverse_socre(decoder_hidden, encoder_outputs)

    def dot_score(self,decoder_hidden, encoder_outputs):
        '''
        :param decoder_hidden: shape:[None, hidden_units]
        :param encoder_outputs: shape:[None, seq_len, hidden_units]
        :return:[None, seq_len]
        '''

        #[None, hidden_units] -> [None, hidden_units,1]
        decoder_hidden_1 = tf.expand_dims(decoder_hidden,-1)

        # [None, seq_len, hidden_units] * [None, hidden_units] -> [None, seq_len, 1]
        y1 = tf.matmul(encoder_outputs, decoder_hidden_1)

        #[None, seq_len, 1] -> [None, seq_len]
        attention_weight = tf.squeeze(y1,axis=-1)
        return attention_weight

    def general_socre(self, decoder_hidden, encoder_outputs):
        '''
        :param decoder_hidden: shape:[None, hidden_units]
        :param encoder_outputs: shape:[None, seq_len, hidden_units]
        :return:
        '''
        # [None, hidden_units] -> [None, hidden_units]
        decoder_hidden_processed = self.w(decoder_hidden)

        # [None, hidden_units] -> [None, hidden_units,1]
        decoder_hidden_processed_1 = tf.expand_dims(decoder_hidden_processed,-1)

        # [None, seq_len, hidden_units] * [None, hidden_units,1] -> [None, seq_len, 1]
        y1 = tf.matmul(encoder_outputs, decoder_hidden_processed_1)

        # [None, seq_len, 1] -> [None, seq_len]
        attention_weight = tf.squeeze(y1, axis=-1)
        return attention_weight

    def concat_socre(self, decoder_hidden, encoder_outputs):
        '''
        :param decoder_hidden: shape:[None, hidden_units]
        :param encoder_outputs: [None, seq_len, hidden_units]
        :return:
        '''
        seq_len = np.shape(encoder_outputs)[1]

        #[None, hidden_units] -> [None, seq_len, hidden_units]
        decoder_hidden_1 = tf.repeat(tf.expand_dims(decoder_hidden,1),seq_len,axis=1)

        #[None, seq_len, hidden_units] -> [None, seq_len, hidden_units*2]
        hc = tf.concat([decoder_hidden_1,encoder_outputs],axis=-1)

        #[None, seq_len, hidden_units*2] -> [None, seq_len, 1] -> [None, seq_len]
        attention_weight = tf.squeeze(self.v(self.w(hc)),axis=-1)

        # [None, seq_len] -> [None, seq_len]
        attention_weight = tf.nn.softmax(attention_weight,axis=-1)

        return attention_weight

    def concat_diverse_socre(self, decoder_hidden, encoder_outputs):
        '''
        :param decoder_hidden: shape:[None, hidden_units]
        :param encoder_outputs: shape: [None, seq_len, hidden_units]
        :return:
        '''
        #[None, hidden_units] -> [None, 1, hidden_units]
        decoder_hidden_1 = tf.expand_dims(decoder_hidden,1)

        #[None, seq_len, hidden_units] + [None, 1, hidden_units] ->[None, seq_len,1] -> [None, seq_len]
        attention_weight = self.v(tf.nn.tanh(self.w1(encoder_outputs)) + self.w2(decoder_hidden_1))
        attention_weight = tf.squeeze(attention_weight,axis=-1)

        #[None, seq_len]->[None, seq_len]
        attention_weight = tf.nn.softmax(attention_weight, axis=-1)
        return attention_weight



if __name__ == '__main__':

    encoder_outputs,state_h = tf.random.normal([2,3,5]),\
                              tf.random.normal([2,5])
    attention_weights = Attention(units=5,method='concat_diverse')(state_h,encoder_outputs)













