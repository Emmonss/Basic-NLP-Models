from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding,LSTM,Bidirectional
import numpy as np
import tensorflow as tf

class Encoder(Model):
    def __init__(self,vocab_size,embeddin_dim,hidden_units,
                 mode='single',method ='concat'):
        super(Encoder,self).__init__()
        assert method in ['sum', 'concat'], "method error"
        assert mode in ['single', 'bio'], "mode error"

        self.mode = mode
        self.method= method
        self.embeddin = Embedding(vocab_size,embeddin_dim,mask_zero=True)
        if self.mode == 'single':
            self.encoder_lstm = LSTM(hidden_units,return_sequences=True,
                                     return_state=True,name='encode_lstm')
        elif self.mode =='bio':
            self.encoder_lstm = Bidirectional(LSTM(hidden_units, return_sequences=True,
                                     return_state=True),merge_mode=method,name = 'encode_lstm')

    def call(self,inputs):
        encoder_embed = self.embeddin(inputs)
        if self.mode == 'single':
            encoder_outputs,state_h,state_c = self.encoder_lstm(encoder_embed)
        else:
            encoder_outputs, state_h_f, state_c_f,state_h_b,state_c_b = self.encoder_lstm(encoder_embed)
            state_h = tf.concat([state_h_f,state_h_b],axis=-1)
            state_c = tf.concat([state_c_f,state_c_b],axis=-1)
        return encoder_outputs, state_h, state_c


if __name__ == '__main__':
    inputs = tf.constant([[2, 5,1],
                         [2, 5,1]])

    encoder_outputs, state_h, state_c = \
        Encoder(vocab_size=1000,embeddin_dim=512,hidden_units=4,mode='bio')(inputs)

    print("encoder_outputs:{}".format(np.shape(encoder_outputs)))
    print("state_h:{}".format(np.shape(state_h)))
    print("state_c:{}".format(np.shape(state_c)))
