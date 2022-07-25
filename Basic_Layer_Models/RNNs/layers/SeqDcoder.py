from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding,LSTM,Dense,Input,GRU
from Basic_Layer_Models.RNNs.layers.SeqAttention import Attention
from Basic_Layer_Models.RNNs.layers.SeqEncoder import EncoderSingleLSTM
import tensorflow as tf
import numpy as np


class Decoder(Model):
    def __init__(self,vocab_size,embedding_dim,dec_units,batch_size=2):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size,embedding_dim)

        self.inputs = Input(shape=(1, embedding_dim), batch_size=batch_size)
        self.lstm_ = LSTM(dec_units,return_sequences=True,
                         return_state=True,stateful=True,
                         name='decode_lstm')
        lstm_out, hidden_state, cell_state = self.lstm_(self.inputs)

        self.lstm = Model(
                inputs=self.inputs,
                outputs=[lstm_out, hidden_state, cell_state])

        self.fc = Dense(vocab_size)
        self.attn = Attention(units=self.dec_units,method='general')


    def call(self,encoder_hidden,encoder_c, target, encoder_outputs,):
        decoder_hidden = encoder_hidden
        print("decoder_hidden:{}".format(np.shape(decoder_hidden)))
        decoder_c = encoder_c

        print(encoder_hidden)
        batch_size = np.shape(encoder_hidden)[0]
        print("batch_size:{}".format(batch_size))

        # decoder_input = tf.constant([[2, 5, 1],[2, 5, 1]])
        # self.lstm.reset_states(states=decoder_hidden)
        decoder_input = tf.transpose(tf.convert_to_tensor([[2 for i in range(batch_size)]]))
        print("decoder_input:{}".format(np.shape(decoder_input)))

        self.forward_step(decoder_input,decoder_hidden,decoder_c,encoder_outputs)

    def forward_step(self, decoder_input, decoder_hidden, decoder_c,encoder_outputs):
        '''
        计算一个时间step的结果
        :param decoder_input:[batch_size,1]
        :param decoder_hidden:[batch_size,hidden_units]
        :param encoder_outputs:[batch_size,encoder_max_len,hidden_units]
        :return:
        '''
        #[batch_size,1] -> [batch_size,1,embed_num]
        decoder_input_embeded = self.embedding(decoder_input)
        print("decoder_input_embeded:{}".format(np.shape(decoder_input_embeded)))
        #[batch_size,1,embed_num] -> [batch_size,1,hidden_units], [batch_size,hidden_units]
        print("decoder_input_init_layer")
        print(self.lstm_.states)
        print('='*50)

        decoder_outputs, decoder_hidden_1, state_c_1 = self.lstm(decoder_input_embeded)
        print("decoder_outputs:{}".format(np.shape(decoder_hidden_1)))
        print(decoder_outputs)
        print("decoder_hidden:{}".format(np.shape(state_c_1)))
        print(self.lstm_.states)


        print('='*50)
        self.lstm_.reset_states(states=[decoder_hidden, decoder_c])
        print("decoder_input_after_layer")
        print(self.lstm_.states)
        #
        decoder_outputs, decoder_hidden, state_c = self.lstm(decoder_input_embeded)
        print("decoder_outputs:{}".format(np.shape(decoder_outputs)))
        print(decoder_outputs)
        print("decoder_hidden:{}".format(np.shape(decoder_hidden)))
        print(self.lstm_.states)
        # #[batch_size,hidden_units],[batch_size,encoder_max_len,hidden_units] -> [batch_size,encoder_max_len]
        # attn_weight = self.attn(decoder_hidden, encoder_outputs)
        #
        # #[batch_size,encoder_max_len] -> [batch_size,1,encoder_max_len]
        # #[batch_size,1,encoder_max_len]* [batch_size,encoder_max_len,hidden_units]->[batch_size,1,hidden_units]
        # context_vector = tf.matmul(tf.expand_dims(attn_weight,axis=1), encoder_outputs)
        #
        # #[batch_size,1,hidden_units] -> [batch_size,hidden_units]
        # context_vector = tf.squeeze(context_vector,axis=1)
        #
        # # out: [batch_size,1,hidden_units] -> [batch_size,hidden_units]
        # # concat: [batch_size,hidden_units] + [batch_size,hidden_units] -> [batch_size,hidden_units*2]
        # # tanh+fc_attn: [batch_size,hidden_units*2] ->[batch_size,hidden_units] -> [batch_size,hidden_units]
        # attention_result = tf.nn.tanh(
        #     self.fc_attn(
        #         tf.concat([context_vector, tf.squeeze(out,axis=1)], axis=-1)
        #     )
        # )

if __name__ == '__main__':
    # encoder_outputs, encoder_hidden = tf.random.normal([2, 10, 20]), \
    #                            tf.random.normal([2, 20])
    target = tf.convert_to_tensor([np.random.randint(1,10) for i in range(10)])

    inputs = tf.constant([[2, 5, 1],
                          [2, 5, 1]])
    print("encoder inputs:{}".format(np.shape(inputs)))
    encoder_model = EncoderSingleLSTM(vocab_size=1000, embeddin_dim=512, hidden_units=4)
    encoder_outputs, encoder_hidden, encoder_c =encoder_model(inputs)
    print("encoder_outputs:{}".format(np.shape(encoder_outputs)))
    print(encoder_outputs)
    print("encoder_hidden:{}".format(np.shape(encoder_hidden)))
    print(encoder_hidden)
    print(encoder_c)
    print('='*100)

    Decoder(vocab_size=1000,embedding_dim=512,dec_units=4)(encoder_hidden,encoder_c,
                                                           target, encoder_outputs,)
    #













