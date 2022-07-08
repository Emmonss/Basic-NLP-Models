from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding,LSTM,Dense
from Basic_Layer_Models.RNNs.layers.SeqAttention import BahdanauAttention

import tensorflow as tf



class Decoder(Model):
    def __init__(self,vocab_size,embedding_dim,dec_units,batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size,embedding_dim)
        self.lstm = LSTM(dec_units,return_sequences=True,return_state=True,name='encode_lstm')
        self.fc = Dense(vocab_size)
        self.attention = BahdanauAttention(dec_units)

    def call(self,y,hidden,enc_output):
        context_vector, attention_weights = self.attention(hidden,enc_output)
        y = self.embedding(y)
        y = tf.concat([tf.expand_dims(context_vector,1),y],axis=1)
        output,state = self.lstm(y)
        output = tf.reshape(output,(-1,output.shape[2]))
        y = self.fc(output)
        return y,state,attention_weights












