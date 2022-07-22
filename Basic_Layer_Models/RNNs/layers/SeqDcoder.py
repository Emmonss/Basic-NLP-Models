from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding,LSTM,Dense,Input
from Basic_Layer_Models.RNNs.layers.SeqAttention import BahdanauAttention
from Basic_Layer_Models.RNNs.layers.SeqEncoder import EncoderSingleLSTM
from Basic_Layer_Models.RNNs.layers.SeqAttention import BahdanauAttention
import tensorflow as tf
import numpy as np


class Decoder(Model):
    def __init__(self,vocab_size,embedding_dim,dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size,embedding_dim)
        self.lstm = LSTM(dec_units,return_sequences=True,return_state=True,name='encode_lstm')
        self.fc = Dense(vocab_size)


    def call(self,y,context_vector, attention_weights):
        y = self.embedding(y)
        print('='*30)
        print(np.shape(context_vector))
        print(np.shape(y))
        y = tf.concat([tf.expand_dims(context_vector,1),y],axis=1)
        output,state = self.lstm(y)
        output = tf.reshape(output,(-1,output.shape[2]))
        y = self.fc(output)
        return y,state,attention_weights

if __name__ == '__main__':
    enc_inputs = Input(shape=(50,), name='inputs')
    encoder_outputs, state_h, state_c = EncoderSingleLSTM(vocab_size=1000,
                                                embeddin_dim=512,
                                                hidden_units=300)(enc_inputs)
    print('=' * 30)
    print(np.shape(encoder_outputs))
    print(np.shape(state_h))

    context_vector, attention_weights = BahdanauAttention(units=300)(state_h, encoder_outputs)
    print('=' * 30)
    print(np.shape(context_vector))
    decoder_outputs, decoder_state, attention_weights = Decoder(vocab_size=1000,
                                                embedding_dim=512,
                                                dec_units=300)(encoder_outputs,
                                                               context_vector,
                                                               attention_weights)
    print(np.shape(decoder_outputs))


    model = Model([enc_inputs,dec_inputs], [decoder_outputs,decoder_state])
    # t.build()
    model.summary()












