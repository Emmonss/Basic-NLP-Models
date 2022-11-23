
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Lambda
from tensorflow_addons.seq2seq.loss import sequence_loss

from Basic_Layer_Models.RNNs.models.NLU_Basic import NLUModel
from Basic_Layer_Models.RNNs.layers.SeqEncoder import Encoder
from Basic_Layer_Models.RNNs.layers.SeqDcoder import Decoder

from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import random

class Seq2SeqAttention(NLUModel):
    def __init__(self,
                 vocab_size,
                 embeding_dim,
                 max_sent_len,
                 encoder_units,
                 lr,
                 opt = 'Adam',
                 encoder_mode = 'bio',
                 encoder_method = 'concat',
                 ):
        super(Seq2SeqAttention, self).__init__()

        self.vocab_size = vocab_size
        self.embeding_dim = embeding_dim
        self.max_sent_len = max_sent_len
        self.encoder_units = encoder_units
        self.encoder_mode = encoder_mode
        self.encoder_method = encoder_method
        if encoder_mode == 'bio' and encoder_method == 'concat':
            self.decoder_units = 2 * self.encoder_units
        else:
            self.decoder_units = self.encoder_units
        self.opt = self.get_opts(opt, lr)

        self.build_model()
        self.compile_model()
    def build_model(self):
        encoder_inputs = Input(shape=(self.max_sent_len,), name='encoder_inputs')
        decoder_inputs = Input(shape=(self.max_sent_len+1,), name='decoder_inputs')
        self.encoder_layer = Encoder(vocab_size=self.vocab_size,
                                embeddin_dim=self.embeding_dim,
                                hidden_units=self.encoder_units,
                                mode=self.encoder_mode,
                                method=self.encoder_method,
                                name='Encoder_Layer')
        self.decoder_layer = Decoder(vocab_size=self.vocab_size,
                                 embedding_dim=self.embeding_dim,
                                 dec_units=self.decoder_units,
                                 seq_maxlen=self.max_sent_len+1,
                                 name='Decoder_Layer')


        encoder_outputs, encoder_states = self.encoder_layer(encoder_inputs)
        decoder_output = self.decoder_layer(decoder_inputs,
                                            encoder_outputs,
                                            encoder_states)
        self.model = Model([encoder_inputs,decoder_inputs], decoder_output)

    def compile_model(self):
        self.model.compile(optimizer=self.opt,
                           loss=self.loss_function,
                           metrics={'Decoder_Layer':'acc'})
        self.model.summary()

    def loss_function(self,y_true, y_pred):
        #target 要去掉<start>标签
        # mask掉<pad>,去除<pad>对于loss的干扰 <pad>默认为0
        mask = tf.cast(tf.math.logical_not(tf.math.equal(y_true, 0)), dtype=tf.float32)
        loss = sequence_loss(logits=y_pred, targets=y_true, weights=mask)
        return loss

    def evaluate(self,inputs):
        encoder_outputs, encoder_states = self.encoder_layer(inputs)
        decoder_outputs, pred_output = self.decoder_layer.evaluate(encoder_outputs, encoder_states)
        return decoder_outputs, pred_output



def simple_sample(vocab_size,seq_len,batch_size=10):
    target=[]
    inputs = []
    for item in range(batch_size):
        target.append([2]+[np.random.randint(3,vocab_size)
                        for i in range(random.randint(int(seq_len/2),seq_len))])
        inputs.append([np.random.randint(3,vocab_size)
                        for i in range(random.randint(int(seq_len/2),seq_len))])
    target = pad_sequences(target, value=0, padding='post', maxlen=seq_len + 1)
    inputs = pad_sequences(inputs, value=0, padding='post', maxlen=seq_len)
    return inputs,target


if __name__ == '__main__':
    seq_len =5
    vocab_size =10
    embed_dim = 100
    hidden_units = 32
    lr = 0.001

    model = Seq2SeqAttention(vocab_size=vocab_size,
                 embeding_dim=embed_dim,
                 max_sent_len=seq_len,
                 encoder_units=hidden_units,
                 lr=lr
                 )

    inputs,target = simple_sample(vocab_size,seq_len)
    # target = np.array([[2, 8, 7, 5, 3],
    #                    [2, 7, 1, 2]])
    # inputs = np.array([[2, 3, 1],
    #                    [4, 3]])


    #
    _, pred_output = model.evaluate(inputs)
    print("init_pred_output:\n{}".format(pred_output))

    #Y是target去掉加上的<start>列，不然计算acc和loss是维度不匹配的
    model.fit_val(X=[inputs,target],Y=target[:,1:],epoch=1000,batch_size=2)

    print("inputs:{}".format(np.shape(inputs)))
    print(inputs)

    print("target:{}".format(np.shape(target[:,1:])))
    print(target[:,1:])

    _, pred_output = model.evaluate(inputs)
    print("trained_pred_output:\n{}".format(pred_output))

    pass
