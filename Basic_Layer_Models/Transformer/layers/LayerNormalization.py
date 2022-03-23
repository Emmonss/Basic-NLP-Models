#utf-8
import os
os.environ['TF_KERAS'] ="1"
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer,Dense
from tensorflow.keras import initializers,activations
from Basic_Layer_Models.Transformer.backend.backend import recompute_grad

class LayerNormalization(Layer):
    def __init__(self,
                 center = True,
                 scale = True,
                 epsilon=None,
                 conditional=False,
                 hidden_units = None,
                 hidden_activation ='linear',
                 hidden_initializer = 'glorot_uniform',
                 **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.epsilon = epsilon or 1e-12
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)
        # print(type(self.hidden_initializer))

    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            mask = mask if mask is not None else []
            mask = [m[None] for m in mask if m is not None]
            if len(mask)==0:
                return None
            else:
                return K.all(K.concatenate(mask,axis=0),axis=0)
        else:
            return mask

    def build(self, input_shape):
        super(LayerNormalization,self).build(input_shape)

        if self.conditional:
            shape = (input_shape[0][-1],)
        else:
            shape = (input_shape[-1],)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,initializer='zeros',name='beta'
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer='ones',name='gamma'
            )

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer
                )

            if self.center:
                self.beta_dense = Dense(
                    units=shape[0],
                    use_bias=False,
                    kernel_initializer='zeros'
                )
            if self.scale:
                self.gamma_dense = Dense(
                    units=shape[0],
                    use_bias=False,
                    kernel_initializer='zeros'
                )
    @recompute_grad
    def call(self,inputs):
        if self.conditional:
            inputs,cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond,1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) +self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = K.mean(outputs,axis=-1,keepdims=True)
            outputs = outputs-mean
        if self.center:
            var = K.mean(K.square(outputs),axis=-1,keepdims=True)
            std = K.sqrt(var+self.epsilon)
            outputs = outputs/std
            outputs = outputs*gamma
        if self.center:
            outputs = outputs+beta

        return outputs

if __name__ == '__main__':
    lr = LayerNormalization()
