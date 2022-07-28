from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,LSTM,Dense,Lambda
from Basic_Layer_Models.RNNs.layers.SeqAttention import Attention
from Basic_Layer_Models.RNNs.layers.SeqEncoder import Encoder
from tensorflow_addons.seq2seq.loss import sequence_loss
import tensorflow as tf
import numpy as np
import random


class Decoder(Model):
    def __init__(self,vocab_size,embedding_dim,
                 dec_units,seq_maxlen,
                 att_mode = 'general',
                 teach_forcing_ran =1,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dec_units = dec_units
        self.seq_maxlen = seq_maxlen
        self.vocab_size = vocab_size

        #teach_forcing 默认每一个batch使用的概率55开,设置为1则训练时全部是teach_forcing
        self.teach_forcing_ran = teach_forcing_ran

        #need layers
        self.embedding = Embedding(self.vocab_size,embedding_dim,
                                   mask_zero=True,name='decoder_embeddings')
        self.lstm = LSTM(dec_units,return_sequences=True,
                         return_state=True,name='decode_lstm')
        self.fc = Dense(vocab_size,name='decoder_attn_fc')
        self.attn = Attention(units=self.dec_units,method=att_mode,name='decoder_attention')


    def call(self,target, encoder_outputs, encoder_states):
        # get init state lstm param from encoder last output
        # [[batch_size,hidden_units],[batch_size,hidden_units]]
        decoder_states = encoder_states

        # give the init decoder input
        #[batch_size, 1]
        decoder_input = tf.expand_dims(target[:, 0], axis=-1)

        all_outputs = []
        if random.random()>self.teach_forcing_ran:
            for t in range(1,self.seq_maxlen):
                # [batch_size,1,vocab_size],[batch_size,hidden_units],[batch_size,hidden_units]
                decoder_outputs_t,decoder_states = \
                    self.forward_step(decoder_input,decoder_states,encoder_outputs)
                all_outputs.append(decoder_outputs_t)
                # [batch_size]
                index = tf.argmax(decoder_outputs_t,axis=-1)
                # [batch_size,1]
                decoder_input = tf.expand_dims(index,-1)
        else:
            for t in range(1,self.seq_maxlen):
                # [batch_size,1,vocab_size],[batch_size,hidden_units],[batch_size,hidden_units]
                decoder_outputs_t,decoder_states = \
                    self.forward_step(decoder_input,decoder_states,encoder_outputs)
                all_outputs.append(decoder_outputs_t)
                #teaching_forcing: 将上一个时间片的target 作为decoder的输入
                # [batch_size,1]
                decoder_input = tf.expand_dims(target[:, t], axis=-1)


        # [seq_maxlen,batch_size,vocab_size] -> [seq_maxlen,batch_size,vocab_size]
        decoder_outputs = Lambda(lambda x: tf.transpose(x,perm=[1,0,2]),name='decoder_trans')(all_outputs)

        #[batch_size,seq_maxlen]
        # pred_output = tf.argmax(decoder_outputs,axis=-1)
        # print(pred_output)

        return decoder_outputs


    def forward_step(self, decoder_input, decoder_states,encoder_outputs):
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
        # print("decoder_input_embeded:{}".format(decoder_input_embeded))


        # [batch_size,1,embed_num] -> [batch_size,1,hidden_units],[batch_size,hidden_units],[batch_size,hidden_units]
        outputs, decoder_hidden, decoder_c = self.lstm(decoder_input_embeded,
                                                       initial_state=decoder_states)

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

        return decoder_outputs_t,[decoder_hidden, decoder_c]

    def evaluate(self, encoder_outputs, encoder_states):
        decoder_states = encoder_states

        #获得输入的batch
        batch_size=np.shape(encoder_outputs)[0]

        # pred 的输入也就只有一维，就是开始的<start>
        decoder_input = tf.expand_dims([2]*batch_size, axis=-1)

        #按照没有teaching_forcing来解码
        all_outputs = []
        for t in range(1, self.seq_maxlen):
            # [batch_size,1,vocab_size],[batch_size,hidden_units],[batch_size,hidden_units]
            decoder_outputs_t, decoder_states = \
                self.forward_step(decoder_input, decoder_states, encoder_outputs)
            all_outputs.append(decoder_outputs_t)
            # [batch_size]
            index = tf.argmax(decoder_outputs_t, axis=-1)
            # [batch_size,1]
            decoder_input = tf.expand_dims(index, -1)
        decoder_outputs = Lambda(lambda x: tf.transpose(x,perm=[1,0,2]),name='decoder_trans')(all_outputs)

        pred_output = tf.argmax(decoder_outputs,axis=-1)

        return decoder_outputs,pred_output



def loss_function(y_true, y_pred):
    # mask掉start,去除start对于loss的干扰
    y_true = y_true[:,1:]
    print(y_true)
    mask = tf.cast(tf.math.logical_not(tf.math.equal(y_true, 0)), dtype=tf.float32)
    loss = sequence_loss(logits=y_pred, targets=y_true, weights=mask)
    return loss


if __name__ == '__main__':
    # encoder_outputs, encoder_hidden = tf.random.normal([2, 10, 20]), \
    #                            tf.random.normal([2, 20])
    seq_len =5
    vocab_size = 10
    embed_dim = 50
    hidden_units = 8

    mode = 'bio'
    #
    # target = np.array([[2]+[np.random.randint(1,vocab_size) for i in range(random.randint(1,seq_len))],
    #                    [2]+[np.random.randint(1,vocab_size) for i in range(random.randint(1,seq_len))]])
    #
    # inputs = np.array([[np.random.randint(1,vocab_size) for i in range(random.randint(1,seq_len))],
    #                    [np.random.randint(1,vocab_size) for i in range(random.randint(1,seq_len))]])

    target = np.array([[2, 8, 7, 5, 3],
                        [3, 7, 1, 2]])
    inputs = np.array([[2, 3, 1],
                        [4, 3]])

    target = pad_sequences(target, value=0,padding='post', maxlen=seq_len+1)
    inputs = pad_sequences(inputs, value=0,padding='post', maxlen=seq_len)


    encoder_model = Encoder(vocab_size=vocab_size,
                            embeddin_dim=embed_dim,
                            hidden_units=hidden_units,
                            mode=mode
                            )

    encoder_outputs, encoder_states =encoder_model(inputs)
    if encoder_model.mode=='bio' and encoder_model.method=='concat':
        decoder_hidden_units = 2 * hidden_units
    else:
        decoder_hidden_units = hidden_units

    decoder_model = Decoder(vocab_size=vocab_size,embedding_dim=embed_dim,
                            dec_units=decoder_hidden_units,
                            seq_maxlen=seq_len+1)
    decoder_output = decoder_model(target,encoder_outputs,encoder_states)

    print("target:{}".format(np.shape(target)))
    print(target)

    print("inputs:{}".format(np.shape(inputs)))
    print(inputs)

    print("decoder_output:{}".format(np.shape(decoder_output)))
    print(decoder_output)

    pred_output = tf.argmax(decoder_output,axis=-1)
    print("pred_output:{}".format(np.shape(pred_output)))
    print(pred_output)

    loss = loss_function(y_true=target,y_pred=decoder_output)
    print("loss:{}".format(loss))

    # sequence_mask = tf.sequence_mask(target, dtype=tf.float32)
    # print(sequence_mask)
    # loss = sequence_loss(logits=pred_output,targets = target, weights=sequence_mask )
    # loss = loss_fuction(target,pred_output)
    # print("loss:{}".format(loss))
    #













