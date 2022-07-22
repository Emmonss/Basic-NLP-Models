from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding,LSTM,Layer,Input
import numpy as np

class EncoderSingleLSTM(Model):
    def __init__(self,vocab_size,embeddin_dim,hidden_units):
        super(EncoderSingleLSTM,self).__init__()

        self.embeddin = Embedding(vocab_size,embeddin_dim,mask_zero=True)

        self.encoder_lstm = LSTM(hidden_units,return_sequences=True,return_state=True,name='encode_lstm')

    def call(self,inputs):
        encoder_embed = self.embeddin(inputs)
        encoder_outputs, state_h, state_c = self.encoder_lstm(encoder_embed)
        return encoder_outputs, state_h, state_c

if __name__ == '__main__':
    inputs = Input(shape=(5,), name='inputs')
    encoder_outputs, state_h, state_c = \
        EncoderSingleLSTM(vocab_size=1000,embeddin_dim=512,hidden_units=300)(inputs)

    model = Model(inputs, (encoder_outputs, state_h, state_c))
    # t.build()
    model.summary()
    #test encoder model componment
    pass