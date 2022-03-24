
import os
import tensorflow as tf
# from bert4keras.layers import ConditionalRandomField
from Basic_Layer_Models.RNNs.layers.ConditionalRandomField import CRF
from tensorflow.keras.layers import Dense,Input,Embedding
from tensorflow.keras.models import Model

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
        inputs = Input(shape=(self.max_len,),name='inputs',dtype=tf.float64)
        embed = Embedding(self.vocab, self.embed, mask_zero=True,dtype=tf.float64)(inputs)
        crf = CRF(self.tag_sum,name='crf',dtype=tf.float64)(embed)
        self.model = Model(inputs, embed)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.model.compile(optimizer=opt,
                            loss={'crf_layer': crf.loss})


if __name__ == '__main__':
    test_model = model_test()
    test_model.model.summary()
    pass