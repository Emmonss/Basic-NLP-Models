import os
os.environ['TF_KERAS'] ="1"
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer,Dense
import tensorflow as tf

from bert4keras import layers
class SinusoidalPositionEmbedding(Layer):
    '''
        Sin-Cos 位置编码
    '''
    def __init__(self,
                 output_dim,
                 merge_mode='add',
                 custom_position_ids=False,
                 **kwargs):
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids


    def call(self,inputs,**kwargs):
        #存在已知的id
        if self.custom_position_ids:
            seq_len = K.shape(inputs)[1]
            inputs,postion_ids = inputs
            if 'float' not in K.dtype(postion_ids):
                position_ids = K.cast(postion_ids,K.floatx())
        #根据序列设置id
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0],input_shape[1]
            position_ids = K.arange(0,seq_len,dtype=K.floatx())[None]


        indices = K.arange(0,self.output_dim//2, dtype=K.floatx())
        indices = K.pow(10000.0 -2*indices/self.output_dim)
        embeddings = tf.einsum('bn,d->bnd',position_ids,indices)
        embeddings = K.stack([K.sin(embeddings),K.cos(embeddings)],axis=-1)
        embeddings = K.reshape(embeddings,(-1,seq_len,self.output_dim))

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * embeddings
        else:
            if not self.custom_position_ids:
                embeddings = K.tile(embeddings, [batch_size,1,1])
            return K.concatenate([inputs,embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul']:
            return input_shape

        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'metge_mode': self.merge_mode,
            'custom_position_ids':self.custom_position_ids
        }
        base_config = super(SinusoidalPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
