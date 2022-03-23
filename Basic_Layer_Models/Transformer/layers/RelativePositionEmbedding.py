import os
os.environ['TF_KERAS'] ="1"
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer,Dense
from Basic_Layer_Models.Transformer.backend.backend import recompute_grad,sequence_masking
import tensorflow as tf

class RelativePositionEmbedding(Layer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer='zeros',
                 **kwargs):
        super(RelativePositionEmbedding,self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = tf.initializers.get(embeddings_initializer)

    def build(self, input_shape):
        super(RelativePositionEmbedding,self).__init__(input_shape)
        self.embeddings = self.add_weight(
            name='embedding',
            shape=(self.input_dim,self.output_dim),
            initializer= self.embeddings_initializer,
        )

    def call(self,inputs):
        pos_ids = self.compute_output_shape(inputs)
        return K.gather(self.embeddings, pos_ids)

    def compute_position_ids(self,inputs):
        q,v = inputs
        #计算位置差
        q_idxs = K.arange(0,K.shape(q)[1],dtype='int32')
        q_idxs = K.expand_dims(q_idxs,1)
        v_idxs = K.arange(0, K.shape(v)[1], dtype='int32')
        v_idxs = K.expand_dims(v_idxs, 0)
        pos_ids = v_idxs - q_idxs
        #后处理操作
        max_position = (self.input_dim - 1) // 2
        pos_ids = K.clip(pos_ids, -max_position, max_position)
        pos_ids = pos_ids + max_position
        return pos_ids

    def compute_output_shape(self, input_shape):
        return (None,None,self.output_dim)

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def get_config(self):
        config = {
            'input_dim':self.input_dim,
            'output_dim':self.output_dim,
            'embeddings_initializer':tf.keras.initializers.serialize(self.embeddings_initializer)
        }
        base_config = super(RelativePositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


