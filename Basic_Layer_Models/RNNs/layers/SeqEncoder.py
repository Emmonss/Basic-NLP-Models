from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding,LSTM

class Encoder(Model):
    def __init__(self,vocab_size,embeddin_dim,hidden_units):
        super(Encoder,self).__init__()

        self.embeddin = Embedding(vocab_size,embeddin_dim,mask_zero=True)
        self.encoder_lstm = LSTM(hidden_units,return_sequences=True,return_state=True,name='encode_lstm')

    def call(self,inputs):
        encoder_embed = self.embeddin(inputs)
        encoder_outputs, state_h, state_c = self.encoder_lstm(encoder_embed)
        return encoder_outputs, state_h, state_c

if __name__ == '__main__':
    #test encoder model componment
    pass