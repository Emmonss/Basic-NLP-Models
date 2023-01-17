import os
os.environ['TF_KERAS'] ="1"
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from BasicLayerModels.Transformer.layers.integerize_shape import integerize_shape

class BiasAdd(Layer):
    @integerize_shape
    def build(self, input_shape):
        super(BiasAdd,self).build(input_shape)
        output_dim = input_shape[-1]
        self.bias = self.add_weight(
            name='bias',
            shape = (output_dim,),
            initializer='zeros',
            trainable=True
        )

    def call(self,inputs):
        return K.bias_add(inputs,self.bias)
