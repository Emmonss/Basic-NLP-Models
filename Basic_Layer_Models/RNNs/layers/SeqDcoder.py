from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding,LSTM,Dense,Input,GRU
from Basic_Layer_Models.RNNs.layers.SeqAttention import Attention
from Basic_Layer_Models.RNNs.layers.SeqEncoder import Encoder
import tensorflow as tf
import numpy as np
import random


class Decoder(Model):
    def __init__(self,vocab_size,embedding_dim,
                 dec_units,seq_maxlen
                 ):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.seq_maxlen = seq_maxlen
        self.vocab_size = vocab_size

        self.embedding = Embedding(self.vocab_size,embedding_dim)

        self.lstm = LSTM(dec_units,return_sequences=True,
                         return_state=True,name='decode_lstm')


        self.fc = Dense(vocab_size)
        self.attn = Attention(units=self.dec_units,method='general')


    def call(self,encoder_hidden,encoder_c, target, encoder_outputs,teach_forcing_ran = -1):

        # get init state lstm param from encoder last output
        # [batch_size,hidden_units]
        decoder_hidden = encoder_hidden
        # [batch_size,hidden_units]
        decoder_c = encoder_c

        batch_size = np.shape(decoder_hidden)[0]
        # give the init decoder input
        # [batch_size,1]
        decoder_input = tf.transpose(tf.convert_to_tensor([[2 for i in range(batch_size)]]))

        # [batch_size,seq_maxlen,vocab_size]
        decoder_output = np.zeros((batch_size,self.seq_maxlen,self.vocab_size))
        if random.random()>teach_forcing_ran:
            for t in range(self.seq_maxlen):
                # [batch_size,1,vocab_size],[batch_size,hidden_units],[batch_size,hidden_units]
                decoder_outputs_t,decoder_hidden, decoder_c = \
                    self.forward_step(decoder_input,decoder_hidden,decoder_c,encoder_outputs)
                decoder_output[:,t,:] = decoder_outputs_t.numpy()
                # [batch_size]
                index = tf.argmax(decoder_outputs_t,axis=-1)
                # [batch_size,1]
                decoder_input = tf.expand_dims(index,-1)

        else:
            for t in range(self.seq_maxlen):
                # [batch_size,1,vocab_size],[batch_size,hidden_units],[batch_size,hidden_units]
                decoder_outputs_t,decoder_hidden, decoder_c = \
                    self.forward_step(decoder_input,decoder_hidden,decoder_c,encoder_outputs)
                decoder_output[:,t,:] = decoder_outputs_t.numpy()
                #teaching  将上一个时间片的target解码的输出层
                # [batch_size]
                index = tf.convert_to_tensor(target[:,t])
                # [batch_size,1]
                decoder_input = tf.expand_dims(index,-1)
        # [batch_size,seq_maxlen,vocab_size]
        decoder_output =tf.convert_to_tensor(decoder_output)

        # [batch_size,seq_maxlen]
        pred_output = tf.argmax(decoder_output,axis=-1)

        return pred_output,decoder_output


    def forward_step(self, decoder_input, decoder_hidden, decoder_c,encoder_outputs):
        '''
        计算一个时间step的结果
        :param decoder_input:[batch_size,1]
        :param decoder_hidden:[batch_size,hidden_units]
        :param encoder_outputs:[batch_size,encoder_max_len,hidden_units]
        :return:decoder_outputs_t:[batch_size,1]
                decoder_hidden:[batch_size,hidden_units]
                decoder_c:[batch_size,hidden_units]
        '''
        # [batch_size,1] -> [batch_size,1,embed_num]
        decoder_input_embeded = self.embedding(decoder_input)

        # reset lstm core state(init from encoder last state params)
        # self.lstm_.reset_states(states=[decoder_hidden, decoder_c])

        # [batch_size,1,embed_num] -> [batch_size,1,hidden_units],[batch_size,hidden_units],[batch_size,hidden_units]
        outputs, decoder_hidden, decoder_c = self.lstm(decoder_input_embeded,
                                                       initial_state=[decoder_hidden, decoder_c])

        # [batch_size,hidden_units],[batch_size,encoder_max_len,hidden_units] -> [batch_size,encoder_max_len]
        attn_weight = self.attn(decoder_hidden, encoder_outputs)

        # [batch_size,encoder_max_len] -> [batch_size,1,encoder_max_len]
        # [batch_size,1,encoder_max_len]* [batch_size,encoder_max_len,hidden_units]->[batch_size,1,hidden_units]
        context_vector = tf.matmul(tf.expand_dims(attn_weight,axis=1), encoder_outputs)

        # [batch_size,1,hidden_units] -> [batch_size,hidden_units]
        context_vector = tf.squeeze(context_vector,axis=1)

        # out: [batch_size,1,hidden_units] -> [batch_size,hidden_units]
        # concat: [batch_size,hidden_units] + [batch_size,hidden_units] -> [batch_size,hidden_units*2]
        # tanh+fc_attn: [batch_size,hidden_units*2] ->[batch_size,hidden_units] -> [batch_size,vocab_size]
        attention_result = tf.nn.tanh(
            self.fc(
                tf.concat([context_vector, tf.squeeze(outputs,axis=1)], axis=-1)
            )
        )

        # [batch_size,vocab_size] -> [batch_size,vocab_size]
        decoder_outputs_t = tf.nn.log_softmax(attention_result,axis=-1)

        return decoder_outputs_t,decoder_hidden, decoder_c


if __name__ == '__main__':
    # encoder_outputs, encoder_hidden = tf.random.normal([2, 10, 20]), \
    #                            tf.random.normal([2, 20])
    seq_len =10
    vocab_size = 100
    embed_dim = 500
    hidden_units = 32

    mode = 'bio'

    target = np.array([[np.random.randint(1,vocab_size) for i in range(seq_len)],
                       [np.random.randint(1,vocab_size) for i in range(seq_len)]])


    inputs = tf.constant([[np.random.randint(1,vocab_size) for i in range(seq_len)],
                       [np.random.randint(1,vocab_size) for i in range(seq_len)]])

    encoder_model = Encoder(vocab_size=vocab_size,
                            embeddin_dim=embed_dim,
                            hidden_units=hidden_units,
                            mode=mode
                            )

    encoder_outputs, encoder_hidden, encoder_c =encoder_model(inputs)

    if encoder_model.mode=='bio' and encoder_model.method=='concat':
        decoder_hidden_units = 2 * hidden_units
    else:
        decoder_hidden_units = hidden_units

    decoder_model = Decoder(vocab_size=vocab_size,embedding_dim=embed_dim,
                            dec_units=decoder_hidden_units,
                            seq_maxlen=seq_len)
    pred_output,decoder_output = decoder_model(encoder_hidden,encoder_c,target,encoder_outputs)

    print("target:{}".format(np.shape(target)))
    print(target)

    print("inputs:{}".format(np.shape(inputs)))
    print(inputs)

    print("pred_output:{}".format(np.shape(pred_output)))
    print(pred_output)
    #













