import os
os.environ['TF_KERAS'] ="1"
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class Concatenate1D(Layer):
    '''
     单层维度拼接
    '''

    def call(self,inputs):
        return K.concatenate(inputs,axis=1)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            mask = []
            for i,m in enumerate(mask):
                if m is None:
                    m = K.ones_like(inputs[i][...,0],dtype='bool')
                mask.append(m)
            return K.concatenate(mask,axis=1)

    def compute_output_shape(self, input_shape):
        if all(shape[1] for shape in input_shape):
            seq_len = sum(shape[1] for shape in input_shape)
            return (input_shape[0][0],seq_len,input_shape[0][2])
        else:
            return (input_shape[0][0],None,input_shape[0][2])



