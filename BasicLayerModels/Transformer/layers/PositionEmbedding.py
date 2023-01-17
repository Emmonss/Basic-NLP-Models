import os
os.environ['TF_KERAS'] ="1"
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer,Dense
from BasicLayerModels.Transformer.backend.backend import recompute_grad,sequence_masking
import tensorflow as tf
from tensorflow.keras import initializers

class PositionEmbedding(Layer):
    """定义位置Embedding，这里的Embedding是可训练的。
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        merge_mode='add',
        hierarchical =None,
        embeddings_initializer='zeros',
        custom_position_ids=False,
        **kwargs
    ):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.hierarchical = hierarchical
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.custom_position_ids = custom_position_ids

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if K.dtype(position_ids) != 'int32':
                position_ids = K.cast(position_ids, 'int32')
            pos_embeddings = K.gather(self.embeddings, position_ids)
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            pos_embeddings = self.embeddings[:seq_len]
            pos_embeddings = K.expand_dims(pos_embeddings, 0)
            if self.merge_mode != 'add':
                pos_embeddings = K.tile(pos_embeddings, [batch_size, 1, 1])

        if self.merge_mode == 'add':
            return inputs + pos_embeddings
        else:
            return K.concatenate([inputs, pos_embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode == 'add':
            return input_shape
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'hierarchical' : self.hierarchical,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer),
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



