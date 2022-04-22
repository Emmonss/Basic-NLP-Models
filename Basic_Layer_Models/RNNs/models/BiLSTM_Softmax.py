from Basic_Layer_Models.RNNs.models.NLU_Basic import NLUModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Embedding,Bidirectional,LSTM
import tensorflow as tf
class BiLSTM(NLUModel):
    def __init__(self,
                 vocab_size,
                 embeddings,
                 hidden_units,
                 tag_num,
                 seg_max_len,
                 lr,
                 opt='Adam'):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embeddings = embeddings
        self.hiden_units = hidden_units
        self.tag_sum = tag_num
        self.seg_max_len = seg_max_len

        self.opt = self.get_opts(opt,lr)

        self.build_model()
        self.compile_model()

    def build_model(self):
        inputs = Input(shape=(self.seg_max_len,),name='inputs')
        embed = Embedding(self.vocab_size, self.embeddings,name='embed')(inputs)
        bi_lstm1 = Bidirectional(LSTM(self.hiden_units, return_sequences=True, activation="tanh"),
                                merge_mode='sum',name='bilistm_1')(embed)
        bi_lstm2 = Bidirectional(LSTM(self.hiden_units, return_sequences=True, activation="tanh"),
                                merge_mode='sum',name='bilistm_2')(bi_lstm1)

        output = Dense(self.tag_sum,activation='softmax',name='outputs')(bi_lstm2)
        output = tf.keras.backend.argmax(output,axis=-1)
        self.model = Model(inputs, output)

    def compile_model(self):
        self.model.compile(optimizer=self.opt,
                           loss={'outputs':'sparse_categorical_crossentropy'},
                           loss_weights={'outputs':1.0},
                           metrics={'outputs':'acc'})
        self.model.summary()

    def fit_val(self,X,Y,valid_data=None,epoch=5,batch_size=32):
        self.history = self.model.fit(X,Y,validation_data=valid_data,epochs=epoch,batch_size=batch_size)

    def fit_train(self, X, Y, val_split=0.1, epoch=5, batch_size=32):
        self.history = self.model.fit(X, Y, validation_split=val_split, epochs=epoch, batch_size=batch_size)


if __name__ == '__main__':
    BiLSTM(vocab_size=5000,
           embeddings=300,
           hidden_units=300,
           seg_max_len=50,
           tag_num=5,
           lr=0.001
           )