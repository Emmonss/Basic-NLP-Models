from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding,LSTM,Bidirectional
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


class Encoder(Model):
    def __init__(self,vocab_size,embeddin_dim,hidden_units,
                 mode='single',method ='concat',**kwargs):
        super(Encoder,self).__init__(**kwargs)
        assert method in ['sum', 'concat'], "method error"
        assert mode in ['single', 'bio'], "mode error"

        self.mode = mode
        self.method= method
        self.embeddin = Embedding(vocab_size,embeddin_dim,mask_zero=True,name='encoder_embeddings')

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
        encoder_states = [state_h,state_c]
        return encoder_outputs, encoder_states


if __name__ == '__main__':
    seq_len = 5
    inputs = np.array([[2, 3, 1],
                       [4, 3]])

    inputs = pad_sequences(inputs, value=0, padding='post', maxlen=seq_len)

    encoder_model = Encoder(vocab_size=10,
                            embeddin_dim=32,
                            hidden_units=4,
                            mode='single'
                            )
    encoder_outputs, encoder_states = encoder_model(inputs)

    print("inputs:{}".format(np.shape(inputs)))
    print("encoder_outputs:{}".format(np.shape(encoder_outputs)))
    print("state_h:{}".format(np.shape(encoder_states[0])))
    print("state_c:{}".format(np.shape(encoder_states[1])))
