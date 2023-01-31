import os
os.environ['TF_KERAS'] ="1"
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer,Dense
from BasicLayerModels.Transformer.backend.backend import recompute_grad,sequence_masking
import tensorflow as tf
from tensorflow.keras import initializers

class MultiHeadAttention(Layer):
    def __init__(self,
                 heads,
                 head_size,
                 out_dim = None,
                 key_size = None,
                 use_bias = True,
                 attention_scale = True,
                 kernel_initializer = 'glorot_uniform',
                 **kwargs):
        super(MultiHeadAttention,self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = out_dim or heads * head_size
        self.key_size = key_size or head_size
        self.use_bias = use_bias,
        self.attention_scale = attention_scale
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        super(MultiHeadAttention,self).build(input_shape)
        self.q_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = Dense(
            units=self.out_dim,
            use_bias = self.use_bias,
            kernel_initializer = self.kernel_initializer
        )

    @recompute_grad
    def call(self,inputs,mask=None,**kwargs):
        q,k,v = inputs[:3]
        q_mask,v_mask = None,None
        if mask is not None:
            if mask[0] is not None:
                q_mask = K.cast(mask[0],K.floatx())
            if mask[2] is not None:
                v_mask = K.cast(mask[2],K.floatx())

        qw = self.q_dense(q)
        vw = self.v_dense(v)
        kw = self.k_dense(v)

        qw = K.reshape(qw,(-1,K.shape(q)[1],self.heads,self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))


        qkv_inputs = [qw,kw,vw]+inputs[3:]
        qv_masks = [q_mask,v_mask]

        #核心计算
        o = self.pay_attention_to(qkv_inputs,qv_masks,**kwargs)

        o = K.reshape(o,(-1,K.shape(o)[1],self.head_size * self.heads))
        o = self.o_dense(o)
        o = sequence_masking(o,q_mask,0)
        return o

    def pay_attention_to(self,inputs,mask=None,**kwargs):
        '''
        实现mutltiheadattention

        o = softmax(Q*K/sqrt(dk)) * V
        :param inputs:
        :param mask:
        :param kwargs:
        :return:o shape(batch,seqlen,heads,head_size)
        '''
        (qw,kw,vw),n = inputs[:3],3
        q_mask,v_mask = mask
        a_bias,p_bias = kwargs.get('a_bias'),kwargs.get('p_bias')
        if a_bias:
            a_bias = inputs[n]
            n+=1
        a = tf.einsum('bjhd,bkhd->bhjk',qw,kw)
        if p_bias == 'typial_relative':
            position_bias = inputs[n]
            a = a+ tf.einsum('bjhd,jkd->bhjk',qw,position_bias)
        elif p_bias == 't5_relative':
            position_bias = K.permute_dimensions(inputs[n],(2,0,1))
            a = a+K.expand_dims(position_bias,0)

        if self.attention_scale:
            a = a / self.key_size**0.5
        if a_bias is not None:
            a = a+ a_bias

        a = sequence_masking(a,v_mask,1,-1)

        a = K.softmax(a)

        o = tf.einsum('bhjk,bkhd->bjhd',a,vw)
        if p_bias == 'typical_relative':
            o = o + tf.einsum('bhjk,jkd->bjhd',a,position_bias)
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],input_shape[0][1],self.out_dim)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[0]

    def get_config(self):
        config = {
            'heads':self.heads,
            'head_size':self.head_size,
            'out_dim':self.out_dim,
            'key_size':self.key_size,
            'use_bias':self.use_bias,
            'attention_scale':self.attention_scale,
            'kernel_initializer':tf.keras.initializers.serialize(self.kernel_initializer)
        }
        base_config = super(MultiHeadAttention,self).get_config()
        return dict(list(base_config.items())+list(config.items()))


if __name__ == '__main__':
    mt = MultiHeadAttention(head_size=256,heads=6)
    



