import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

class BahdanauAttention(Model):
    def __init__(self,units):
        super(BahdanauAttention, self).__init__()
        self.w1 = Dense(units)
        self.w2 = Dense(units)
        self.v = Dense(1)

    def call(self,query,values):
        hidden_with_time_axis = tf.expand_dims(query,1)
        score = self.v(tf.nn.tanh(self.w1(values)+self.w2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector,axis=1)
        return context_vector,attention_weights












