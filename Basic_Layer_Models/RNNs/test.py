
import os
os.environ['TF_KERAS'] ="1"
import tensorflow as tf
# from bert4keras.layers import ConditionalRandomField


from Basic_Layer_Models.RNNs.layers.ConditionalRandomField import CRF
from tensorflow.keras.layers import Dense,Input,Embedding
from tensorflow.keras.models import Model,Sequential
os.environ['CUDA_VISIBLE_DEVICES'] = " "

class model_test():
    def __init__(self):
        self.model = None
        self.crf_lr_multiplier = 1000
        self.max_len = 100
        self.tag_sum = 5
        self.vocab = 3000
        self.embed = 300
        self.load_model()


    def load_model(self):
        myModel = Sequential()
        myModel.add(tf.keras.layers.Input(shape=(self.max_len,)))
        myModel.add(tf.keras.layers.Embedding(self.vocab, self.embed))
        myModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            self.tag_sum, return_sequences=True, activation="tanh"), merge_mode='sum'))
        myModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            self.tag_sum, return_sequences=True, activation="softmax"), merge_mode='sum'))
        crf = CRF(self.tag_sum, name='crf_layer')
        myModel.add(crf)
        myModel.compile('adam', loss={'crf_layer': crf.get_loss})
        self.model = myModel

        # inputs = Input(shape=(self.max_len,),name='inputs')
        # embed = Embedding(self.vocab, self.embed)(inputs)
        # crf = CRF(self.tag_sum,name='crf')(embed)
        # self.model = Model(inputs, crf)
        # opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        # self.model.compile(optimizer=opt,
        #                     loss='binary_crossentropy')
        # self.model.compile(optimizer=opt,
        #                     loss={'crf_layer': crf.loss})


if __name__ == '__main__':
    test_model = model_test()
    test_model.model.summary()
    pass